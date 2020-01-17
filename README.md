# caculate_ap

### 介绍
计算准确率、漏检、错检

### 数据准备
- 预测结果文件格式  
```
# index
image_name
box_num
label left_top_x left_top_y right_bottom_x right_bottom_y score

例如：
# 0
YanHuo00006.jpg
1
7 1941.0 977.7 2053.3 1178.3 0.9106
# 1
YanHuo00020.jpg
2
7 304.2 580.0 858.9 714.8 0.9550
7 841.6 641.5 931.2 698.6 0.8103
```
- 标注信息：标准xml格式


### 准确率计算
- 修改config.json
```
{
    // 标注路径
    "anns_folder": "YanHuo/xml",
    // nms阈值
    "overlap": 0.5,
    // 预测的结果文件
    "results_file": "YanHuo/tongdao_test_vgg16_YanHuo.frcnn",
    // 图片路径
    "imgs_folder": "YanHuo/jpg",
    // 结果图片保存k路径
    "out_folder": "result",
    // 类别
    "classes":["normal","abnormal","fuzzy"]
}
```
- run
```
python caculate_acc.py
```

