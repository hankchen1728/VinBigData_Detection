from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config = './cfg.py' 
checkpoint = './tools/x101/epoch_12.pth'

model = init_detector(config, checkpoint, device='cuda:0')

img = '/work/VinBigData/preprocessed/test/img_npz/ffccf1709d0081d122a1d1f9edbefdf1.npz'
result = inference_detector(model, img)

show_result_pyplot(model, img, result)
