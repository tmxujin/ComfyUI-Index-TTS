# 字幕处理节点使用指南

本项目新增了两个用于字幕处理的节点，支持SRT和结构化JSON格式的字幕处理，特别适用于数字人多角色对话场景。

## 节点介绍

### 1. Enhanced Subtitle Batch Iterator (增强版字幕批处理迭代器)

这是一个功能强大的字幕处理节点，支持以下特性：

- 支持SRT格式字幕（自动解析为角色0）
- 支持结构化JSON字幕（支持多角色）
- 旁白自动标注为角色0
- 输出字幕文本供后期使用
- 整合解析和迭代功能于一体
- 支持批量输出模式，ComfyUI会自动对每个片段执行一次工作流
- 新增total_iterations输出，方便循环控制
- 支持多种时间格式解析（HH:MM:SS.mmm 和 M:SS.mmm）
- 支持多种字幕字段名称（字幕、text、subtitle）

#### 输入参数

- `subtitle_text`: 字幕文本内容（STRING类型）
- `subtitle_type`: 字幕类型（"auto", "srt", "json"，默认为"auto"）
- `reset_iterator`: 是否重置迭代器（BOOLEAN类型，默认为False）
- `narrator_as_character_0`: 是否将旁白标记为角色0（BOOLEAN类型，默认为True）

#### 输出参数

- `mask_index`: 角色对应的mask索引（INT类型，列表形式）
- `character_id`: 角色ID（STRING类型，列表形式）
- `start_time`: 字幕开始时间（FLOAT类型，列表形式）
- `end_time`: 字幕结束时间（FLOAT类型，列表形式）
- `subtitle_text`: 字幕文本内容（STRING类型，列表形式）
- `duration`: 字幕时长（FLOAT类型，列表形式，单位：秒，精确到小数点后1位，采用"只入不舍"方式处理，确保视频长度足够）
- `total_iterations`: 总字幕条数（INT类型，单个值）

#### 使用示例

##### JSON格式字幕示例：

支持两种JSON格式：

1. 包含segments键的对象格式：

```json
{
  "segments": [
    {"id": "Narrator", "start": "0:00.000", "end": "0:05.300", "字幕": "这是一个旁白片段"},
    {"id": "Character1", "start": "0:05.300", "end": "0:10.700", "字幕": "这是角色1的台词"},
    {"id": "Character2", "start": "0:10.700", "end": "0:15.200", "字幕": "这是角色2的台词"}
  ]
}
```

2. 直接的数组格式：

```json
[
  {"id": "Narrator", "start": "0:00.000", "end": "0:05.300", "字幕": "这是一个旁白片段"},
  {"id": "Character1", "start": "0:05.300", "end": "0:10.700", "字幕": "这是角色1的台词"},
  {"id": "Character2", "start": "0:10.700", "end": "0:15.200", "字幕": "这是角色2的台词"}
]
```

#### 输出示例

对于上述示例字幕，节点将输出：

- `mask_index`: [0, 1, 2]
- `character_id`: ["0", "Character1", "Character2"]
- `start_time`: [0.0, 5.3, 10.7]
- `end_time`: [5.3, 10.7, 15.2]
- `subtitle_text`: ["这是一个旁白片段", "这是角色1的台词", "这是角色2的台词"]
- `duration`: [5.3, 5.4, 4.5] (时长 = 结束时间 - 开始时间，精确到小数点后1位，采用"只入不舍"方式处理)
- `total_iterations`: 3

##### SRT格式字幕示例：

```
1
00:00:00,000 --> 00:00:05,000
这是一个旁白片段

2
00:00:05,000 --> 00:00:10,000
这是角色1的台词

3
00:00:10,000 --> 00:00:15,000
这是角色2的台词
```

### 2. Subtitle Text Formatter (字幕文本格式化器)

用于将字幕文本格式化为可用于视频叠加的格式。

#### 输入参数

- `subtitle_text`: 字幕文本内容（STRING类型，强制输入）
- `start_time`: 开始时间（FLOAT类型，强制输入）
- `end_time`: 结束时间（FLOAT类型，强制输入）
- `include_timeline`: 是否包含时间轴（BOOLEAN类型，默认为True）

#### 输出参数

- `formatted_text`: 格式化后的文本（STRING类型）

## 批量处理模式

Enhanced Subtitle Batch Iterator节点现在支持批量处理模式，该模式具有以下优势：

1. 一次性解析所有片段
2. 返回列表而不是单个值
3. ComfyUI会自动对每个元素执行一次工作流
4. 新增total_iterations输出，方便循环节点使用

### 工作流连接方式

#### 方式A：直接批量处理（最简单）✅✅✅

```
[EnhancedSubtitleBatchIterator]
    ↓ mask_index (列表: [1, 2, 3, 1, 2])
[ImageMaskSwitch] 
    ↓ (ComfyUI自动执行5次)
[ImageResize]
    ↓
[SaveImage] (自动保存5张图片)
```

ComfyUI行为：

- 检测到mask_index是列表 [1, 2, 3, 1, 2]
- 自动执行5次工作流
  - 第1次：select=1
  - 第2次：select=2
  - 第3次：select=3
  - ...

#### 方式B：配合循环节点（如果需要更多控制）

```
[EnhancedSubtitleBatchIterator]
    ↓ total_iterations (单个整数: 5)
[IteratorOpen] ← 连接total_iterations
    ↓
[IteratorCounter]
    ↓ current_index
[根据索引处理...]
    ↓
[IteratorClose]
```

## 使用场景

### 数字人多角色对话

在数字人项目中，可以通过以下步骤使用这些节点：

1. 准备结构化的字幕数据（JSON格式）
2. 使用Enhanced Subtitle Batch Iterator节点解析字幕并逐条输出
3. 根据角色ID和mask索引切换不同的数字人模型和遮罩
4. 结合TTS节点生成对应角色的语音
5. 使用Subtitle Text Formatter节点格式化字幕文本用于视频叠加

### 工作流示例

项目中提供了以下示例工作流文件：

1. `workflow/数字人字幕处理示例.json` - 展示了如何使用这些节点
2. `workflow/数字人批量处理循环工作流.json` - 展示了如何使用批量处理模式

### 执行示例

#### 输入字幕

```json
{
  "segments": [
    {"id": "Character1", "start": "0:00.000", "end": "0:03.000"},
    {"id": "Character2", "start": "0:03.000", "end": "0:06.000"},
    {"id": "Character3", "start": "0:06.000", "end": "0:09.000"},
    {"id": "Character4", "start": "0:09.000", "end": "0:12.000"}
  ]
}
```

#### 节点输出

```
mask_index = [1, 2, 3, 4]
character_id = ["Character1", "Character2", "Character3", "Character4"]
start_time = [0.0, 3.0, 6.0, 9.0]
end_time = [3.0, 6.0, 9.0, 12.0]
duration = [3.0, 3.0, 3.0, 12.0] (时长 = 结束时间 - 开始时间，采用"只入不舍"方式处理)
total_iterations = 4
```

#### ComfyUI执行

1. 第1次执行:
   - ImageMaskSwitch.select = 1 → 输出Character1图片
   - SaveImage → ComfyUI_Character1_00001.png

2. 第2次执行:
   - ImageMaskSwitch.select = 2 → 输出Character2图片
   - SaveImage → ComfyUI_Character1_00002.png

3. 第3次执行:
   - ImageMaskSwitch.select = 3 → 输出Character3图片
   - SaveImage → ComfyUI_Character1_00003.png

4. 第4次执行:
   - ImageMaskSwitch.select = 4 → 输出Character4图片
   - SaveImage → ComfyUI_Character1_00004.png

## 立即测试

1. 替换代码 - 覆盖subtitle_processor.py
2. 重启ComfyUI
3. 打开工作流 - 不需要修改连接
4. Queue Prompt - 自动批处理所有片段

完成！🎉

## 重要注意事项

### 内存占用

- 所有片段会同时加载到内存
- 如果片段数>100，建议分批处理或使用Python脚本

### SaveImage节点

- 会自动为每个批次生成不同文件名
- 格式：prefix_00001.png, prefix_00002.png, ...

### 调试输出

节点会在控制台打印详细信息：
```
✅ 批量模式启动: 共解析到 4 个片段
  [1/4] 角色'Character1' (mask=1) | 0.000s - 3.000s | 这是角色1的台词...
  [2/4] 角色'Character2' (mask=2) | 3.000s - 6.000s | 这是角色2的台词...
  [3/4] 角色'Character3' (mask=3) | 6.000s - 9.000s | 这是角色3的台词...
  [4/4] 角色'Character4' (mask=4) | 9.000s - 12.000s | 这是角色4的台词...
```

## 立即测试

1. 替换代码 - 覆盖subtitle_processor.py
2. 重启ComfyUI
3. 打开工作流 - 不需要修改连接
4. Queue Prompt - 自动批处理所有片段

完成！🎉

## 注意事项

1. JSON格式支持多种字段名称，如"id"和"character_id"都可识别为角色ID
2. 时间格式支持"M:SS.mmm"格式，如"1:23.456"
3. 旁白角色可以使用"Narrator"或"旁白"标识
4. 字幕文本字段支持"字幕"、"text"、"subtitle"等多种名称
5. 节点具有状态保持功能，适用于批处理场景