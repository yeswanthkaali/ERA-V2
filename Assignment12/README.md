# Session 7 Assignment

This assignment is for training Yolov3 for custome dataset here we have picked a dataset of superman,batman and ironman
## Training Log
 YoloV3 % python train.py --data data/customdata/custom.data --batch 10 --cache --cfg cfg/yolov3-custom.cfg --epochs 3 --nosave

Namespace(accumulate=4, adam=False, batch_size=10, bucket='', cache_images=True, cfg='cfg/yolov3-custom.cfg', data='data/customdata/custom.data', device='', epochs=3, evolve=False, img_size=[512], multi_scale=False, name='', nosave=True, notest=False, rect=False, resume=False, single_cls=False, weights='weights/yolov3-spp-ultralytics.pt')
Using CPU

WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
[INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
[INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm2d'>.
[INFO] Register count_relu() for <class 'torch.nn.modules.activation.LeakyReLU'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
[INFO] Register zero_ops() for <class 'torch.nn.modules.pooling.MaxPool2d'>.
[INFO] Register count_upsample() for <class 'torch.nn.modules.upsampling.Upsample'>.
/opt/anaconda3/envs/yolo/lib/python3.8/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:3550.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 225 layers, 6.25841e+07 parameters, 6.25841e+07 gradients, 117.1 GFLOPS
Caching labels (87 found, 27 missing, 1 empty, 0 duplicate, for 115 images): 100%|█| 115/115 [00:00<00:00, 3692.31it/
Caching images (0.1GB): 100%|█████████████████████████████████████████████████████| 115/115 [00:00<00:00, 144.04it/s]
Caching labels (87 found, 27 missing, 1 empty, 0 duplicate, for 115 images): 100%|█| 115/115 [00:00<00:00, 24306.84it
Caching images (0.0GB): 100%|██████████████████████████████████████████████████████| 115/115 [00:01<00:00, 79.74it/s]
Image sizes 512 - 512 train, 512 test
Using 8 dataloader workers
Starting training for 3 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
       0/2        0G       5.7      64.7      2.32      72.7        12       512: 100%|█| 12/12 [02:57<00:00, 14.79s/
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█| 12/12 [04:43<00:00, 23.59s/
                 all       115       139  0.000302      0.02  0.000399  0.000594

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
       1/2        0G      5.21      4.71      2.28      12.2        17       512: 100%|█| 12/12 [03:06<00:00, 15.54s/
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█| 12/12 [02:44<00:00, 13.70s/
                 all       115       139         0         0   0.00321         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
       2/2        0G      4.46      2.89      2.23      9.57        10       512: 100%|█| 12/12 [02:57<00:00, 14.81s/
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█| 12/12 [01:38<00:00,  8.24s/
                 all       115       139         0         0    0.0408         0
3 epochs completed in 0.316 hours.


## Test Output
![Group Norm](test_batch0.png)




**Yeswanth**
