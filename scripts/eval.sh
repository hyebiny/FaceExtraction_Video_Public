#!/usr/bin/env bash
# CJENM-test CelebA-HQ-WO-test test_benchmark_02


root='/logs/stage1/231008_232647/test_images/' # test_benchmark_02'
python evaluation.py --pred-dir '.'$root \
                --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/2023-09-21-01-46-07/test_images_20/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/2023-09-19-21-29-10/test_images_40/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/2023-09-14-12-22-40/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/test/2023-09-14-12-23-12/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/test/2023-09-15-00-31-17/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'

# root='/experiments/test/2023-09-16-10-11-12/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'
# root='/experiments/test/2023-09-16-10-11-25/test_images/test_benchmark_02'
# python evaluation.py --pred-dir '.'$root \
#                 --label-dir '/home/jhb/dataset/FM_dataset/test_benchmark_02'


