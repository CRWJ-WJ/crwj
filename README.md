# 基于计算机视觉与机器学习的金属零部件表面缺陷智能检测与分级系统
> 课程设计项目：B/S架构工业缺陷检测原型系统
技术栈：FastAPI + OpenCV‑Python + PyTorch‑CNN + 遗传算法 + SQLite + HTML/JS

## 项目简介
本系统实现金属零部件表面缺陷图片上传、图像预处理、缺陷检测识别、缺陷等级判定、参数自适应优化、检测结果可视化、检测报告生成、历史记录存储功能。
1. OpenCV实现图像灰度化、滤波降噪、ROI感兴趣区域提取
2. CNN卷积神经网络完成划痕、夹杂、斑块等金属表面缺陷分类识别
3. 遗传算法实现视觉检测阈值自适应寻优，完成参数闭环优化
4. FastAPI搭建B/S后端服务，SQLite持久化保存全部检测记录
5. Web前端完成图片上传、缺陷标注可视化、历史记录查询、算法参数配置

## 数据集来源

### 主数据集：NEU‑DET东北大学热轧带钢表面缺陷数据集
- 获取方式：公开工业缺陷数据集  http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm
- 数据集内容：共1800张灰度工业工件图像；6类缺陷：`crazing(裂纹)、inclusion(夹杂)、patches(斑块)、pitted_surface(点蚀)、rolled‑in_scale(氧化皮)、scratches(划痕)`
- 目录结构：
