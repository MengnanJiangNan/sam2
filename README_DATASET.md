# SAM2 数据集说明

本目录包含SAM2项目使用的两个重要数据集，已压缩为tar.gz格式便于下载和传输。

## 数据集文件

### 1. PartObjaverse-Tiny.tar.gz (920MB)
- **原始大小**: 1.2GB
- **压缩后大小**: 920MB
- **内容**: PartObjaverse-Tiny数据集
- **用途**: 用于SAM2模型的训练和测试，包含3D物体的分割标注数据
- **位置**: `sam2/dataset/PartObjaverse-Tiny.tar.gz`

### 2. render.tar.gz (77GB)
- **原始大小**: 79GB
- **压缩后大小**: 77GB
- **内容**: 渲染数据集
- **用途**: 包含3D物体的渲染图像，用于视觉理解和分割任务
- **位置**: `sam2/dataset/render.tar.gz`

## 解压说明

### 解压PartObjaverse-Tiny数据集
```bash
cd sam2/dataset
tar -xzf PartObjaverse-Tiny.tar.gz
```

### 解压render数据集
```bash
cd sam2/dataset
tar -xzf render.tar.gz
```

## 数据集结构

解压后的目录结构：
```
sam2/dataset/
├── PartObjaverse-Tiny/
│   └── [3D物体分割数据]
├── render/
│   └── [渲染图像数据]
└── [其他数据集文件]
```

## 注意事项

1. **存储空间**: render数据集解压后需要约79GB存储空间，请确保有足够的磁盘空间
2. **下载时间**: render.tar.gz文件较大(77GB)，下载可能需要较长时间，建议使用支持断点续传的下载工具
3. **解压时间**: 大文件解压可能需要较长时间，请耐心等待

## 使用建议

- 如果只需要进行基础测试，可以先下载PartObjaverse-Tiny.tar.gz
- 如果需要完整的训练功能，建议下载两个数据集文件
- 建议在SSD上解压以提高I/O性能

## 备份位置

- **公司电脑备份**: 这两个压缩文件在公司电脑的对应备份文件夹下也有保存
- **本地位置**: 当前文件位于本地开发环境

## 创建时间

- 压缩文件创建时间: 2024年8月6日
- 数据集版本: SAM2项目专用版本 