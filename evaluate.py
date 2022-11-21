import os 
import argparse

from mdou.evaluation.simulation import  *

#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_agent_type', type=str,
            default='DeepAgent')

    parser.add_argument('--landlord_up', type=str,
            default='baselines/universal_ADP/UniversalModel.ckpt')
    parser.add_argument('--landlord_up_agent_type', type=str,
            default='UniversalAgent')

    parser.add_argument('--landlord_down', type=str,
            default='baselines/universal_ADP/UniversalModel.ckpt')
    parser.add_argument('--landlord_down_agent_type', type=str,
            default='UniversalAgent')

    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--evaluate_device_cpu', action='store_true',
                        help='Use CPU as actor device')
    parser.add_argument('--logs', type=str, default='logs.csv')

    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    #for base in baselines:
    evaluate(args)


