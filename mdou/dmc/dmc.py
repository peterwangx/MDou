import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np
import ctypes
import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act

mean_episode_return_buf = deque(maxlen=100)

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss

def learn(actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)

    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf.append(torch.mean(episode_returns).to(device))
        
    with lock:
        learner_outputs = model(obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return': torch.mean(torch.stack([_r for _r in mean_episode_return_buf])).item(),
            'loss': loss.item(),
        }
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model().load_state_dict(model.state_dict())

        return stats

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
        
    for device in device_iterator:
        _free_queue = ctx.SimpleQueue()
        _full_queue = ctx.SimpleQueue()
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizer = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return',
        'loss',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = 0

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )
        #for k in ['landlord', 'landlord_up', 'landlord_down']:
        learner_model.get_model().load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        for device in device_iterator:
            models[device].get_model().load_state_dict(learner_model.get_model().state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    cur_frame = mp.Value(ctypes.c_uint64, 1)
    # Starting actor processes
    for device in device_iterator:
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags, cur_frame))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, local_lock, position_lock, cur_frame, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device], full_queue[device], buffers[device], flags, local_lock)
            _stats = learn(models, learner_model.get_model(), batch,
                optimizer, flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                cur_frame.value = frames
                position_frames += T * B


    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] =  threading.Lock()
    position_locks = threading.Lock()

    for device in device_iterator:
        for i in range(flags.num_threads):
            thread = threading.Thread(
                target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,device,locks[device],position_locks,cur_frame))
            thread.start()
            threads.append(thread)
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': learner_model.get_model().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        model_weights_dir = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid, 'UniversalModel'+'_weights_'+str(frames)+'.ckpt')))
        torch.save(learner_model.get_model().state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = position_frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = (position_frames-position_start_frames)/(end_time-start_time)
            log.info('After %i (%i) frames: @ %.1f fps (avg@ %.1f fps) (%.1f) Stats:\n%s',
                     frames,
                     position_frames,
                     fps,
                     fps_avg,
                     position_fps,
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
