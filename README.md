# 车牌识别
对车牌的文字+颜色进行识别，识别颜色通过对图片的主颜色进行K-Means聚类得到，具体可以看`model.get_color.py`。

这里感谢[https://github.com/we0091234/crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition/tree/plate_color)的工作。



## 环境



## 数据
1. 图片命名规则：车牌号_序号.jpg；
2. 然后执行`create_label_txt.py`，用来生成`[img_path \t gtlabel]`的txt文件；
   ```
   python plateLabel.py --image_path datasets/train/ --label_file datasets/train.txt
   python plateLabel.py --image_path datasets/val/ --label_file datasets/val.txt
   ```
   ```
   datasets/val/云A390WG_0.jpg 25 52 45 51 42 72 58 
   datasets/val/云A3HG75_0.jpg 25 52 45 59 58 49 47 
   datasets/val/云A566FD_0.jpg 25 52 47 48 48 57 55 
   datasets/val/云A5A691_0.jpg 25 52 47 52 48 51 43
   ```
3. 将`train.txt`和`val.txt`路径写入`360CC_config.yaml`文件中。
   ```
   TXT_FILE: {'train': 'datasets/train.txt', 'val': 'datasets/val.txt'}
   ```


## 训练

```
   python train.py --cfg configs/360CC_config.yaml
```
结果保存在`output`文件夹下。


## 推理
- 支持单张图片推理；
   ```
   python inference.py --cfg configs/360CC_config.yaml --image_path 'datasets/val/云Axxxxx_0.jpg'
   ```
  
    **IMAGE:** 

    ![Image](test_images/1.jpg)

    **RESULT:** 

    ![Image](test_images/1res.jpg)

- 也支持整个文件夹下所有图片推理；
   ```
   python inference.py --cfg configs/360CC_config.yaml --image_path 'datasets/val'
   ```


## 测试
- 对某个文件夹下的所有图片`[云A0E9H2_0.jpg, ...]`车牌字进行测试，得到准确率accuracy。
   ```
  python val.py --cfg configs/360CC_config.yaml --image_path 'datasets/val'
  ```

## 导出onnx

```
    python export.py --weights saved_model/best.pth --save_path saved_model/best.onnx --simplify
```

#### onnx 推理

```
python onnx_infer.py --onnx_file saved_model/best.onnx  --image_path images/test.jpg
```


