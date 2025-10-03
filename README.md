# 免责声明

本项目基于B站开源项目进行二次开发，由本人对项目进行了ComfyUI的实现，并进行了部分功能优化与调整与进阶功能的开发。然而，需要强调的是，本项目严禁用于任何非法目的以及与侵犯版权相关的任何行为！本项目仅用于开源社区内的交流与学习，以促进技术共享与创新，旨在为开发者提供有益的参考和学习资源。

在此郑重声明，本项目所有个人使用行为与开发者本人及本项目本身均无任何关联。开发者对于项目使用者的行为不承担任何责任，使用者应自行承担使用过程中可能产生的所有风险和法律责任。请广大使用者在遵守法律法规及相关规定的前提下，合理、合法地使用本项目，维护开源社区的良好秩序与健康发展。

感谢您的理解与支持！


# ComfyUI-Index-TTS

使用IndexTTS模型在ComfyUI中实现高质量文本到语音转换的自定义节点。支持中文和英文文本，可以基于参考音频复刻声音特征。

![示例截图1](https://github.com/user-attachments/assets/41960425-f739-4496-9520-8f9cae34ff51)
![示例截图2](https://github.com/user-attachments/assets/1ff0d1d0-7a04-4d91-9d53-cd119250ed67)
![微信截图_20250605215845](https://github.com/user-attachments/assets/d5eb22f6-2ca2-40cf-a619-d709746f83e3)



## 功能特点

- 支持中文和英文文本合成
- 基于参考音频复刻声音特征（变声功能）
- 支持调节语速（原版不支持后处理实现效果会有一点折损）
- 多种音频合成参数控制
- Windows兼容（无需额外依赖）


## 废话两句

- 生成的很快，真的很快！而且竟然也很像！！！
- 效果很好，感谢小破站的开源哈哈哈哈哈 
- 如果你想体验一下效果 附赠道友B站的传送阵[demo](https://huggingface.co/spaces/IndexTeam/IndexTTS)
- 如果你不知道去哪找音频，那我建议你去[隔壁](https://drive.google.com/drive/folders/1AyB3egmr0hAKp0CScI0eXJaUdVccArGB)偷哈哈哈哈哈

## 演示案例

以下是一些实际使用效果演示：

| 参考音频 | 输入文本 | 推理结果 |
|---------|---------|---------|
| <video src="https://github.com/user-attachments/assets/5e8cb570-242f-4a16-8472-8a64a23183fb"></video> | 我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。  我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。 | <video src="https://github.com/user-attachments/assets/d8b89db3-5cf5-406f-b930-fa75d13ff0bd"></video> |
| <video src="https://github.com/user-attachments/assets/8e774223-e0f7-410b-ae4e-e46215e47e96"></video> | 我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。 | <video src="https://github.com/user-attachments/assets/6e3e63ed-2d3d-4d5a-bc2e-b42530748fa0"></video> |

- 长文本测试：

<video src="https://github.com/user-attachments/assets/6bfa35dc-1a30-4da0-a4dc-ac3def25452b"></video>

- 多角色小说测试：

<video src="https://github.com/user-attachments/assets/6d4737f4-9d75-431e-bb11-fe3e86a4ab0e"></video>



## 更新日志

### 2025-06-24

- pro节点新增了对于字幕的json输出，感谢@qy8502提供的玩法思路

![image](https://github.com/user-attachments/assets/e7f5e92a-7f76-48a1-ba01-86143d10d359)


### 2025-06-05

- 改进了小说文本解析器（Novel Text Parser）的功能
  - 增加了对预格式化文本的检测和处理
  - 优化了对话检测和角色识别算法
  - 改进了中文角色名称的识别
  - 支持引号中的对话自动识别

## 多角色小说文本解析

本项目包含一个专门用于解析小说文本的节点（Novel Text Structure Node），可以将普通小说文本解析为多角色对话结构，以便生成更加自然的多声音TTS效果。

### 使用说明

- 节点会尝试自动识别小说中的角色对话和旁白部分
- 对话部分会标记为`<CharacterX>`形式（X为数字，最多支持5个角色）
- 旁白部分会标记为`<Narrator>`
- 解析后的文本可直接用于多声音TTS生成

### 局限性

- 当前解析算法并不完美，复杂的小说结构可能导致错误的角色识别
- 对于重要文本，建议使用LLM（如GPT等）手动拆分文本为以下格式：

```
<Narrator>少女此时就站在院墙那边，她有一双杏眼，怯怯弱弱。</Narrator>
<Narrator>院门那边，有个嗓音说：</Narrator>
<Character1>"你这婢女卖不卖？"</Character1>
<Narrator>宋集薪愣了愣，循着声音转头望去，是个眉眼含笑的锦衣少年，站在院外，一张全然陌生的面孔。</Narrator>
<Narrator>锦衣少年身边站着一位身材高大的老者，面容白皙，脸色和蔼，轻轻眯眼打量着两座毗邻院落的少年少女。</Narrator>
<Narrator>老者的视线在陈平安一扫而过，并无停滞，但是在宋集薪和婢女身上，多有停留，笑意渐渐浓郁。</Narrator>
<Narrator>宋集薪斜眼道：</Narrator>
<Character2>"卖！怎么不卖！"</Character2>
<Narrator>那少年微笑道：</Narrator>
<Character1>"那你说个价。"</Character1>
<Narrator>少女瞪大眼眸，满脸匪夷所思，像一头惊慌失措的年幼麋鹿。</Narrator>
<Narrator>宋集薪翻了个白眼，伸出一根手指，晃了晃，</Narrator>
<Character2>"白银一万两！"</Character2>
<Narrator>锦衣少年脸色如常，点头道：</Narrator>
<Character1>"好。"</Character1>
<Narrator>宋集薪见那少年不像是开玩笑的样子，连忙改口道：</Narrator>
<Character2>"是黄金万两！"</Character2>
<Narrator>锦衣少年嘴角翘起，道：</Narrator>
<Character1>"逗你玩的。"</Character1>
<Narrator>宋集薪脸色阴沉。</Narrator>
```

### 示例用法

1. 将小说文本输入到 Novel Text Structure 节点
2. 连接输出到 Index TTS Pro 节点
3. 设置不同角色的语音
4. 运行工作流生成多声音小说朗读
5. 实在不会看我最新增加的工作流
6. 如果你想在comfyui中一站式完成这个，我推荐你使用各类的llm节点，比如[kimichat](https://github.com/chenpipi0807/PIP_KIMI2comfyui)
7. 我也提供了一段llm提示词模板，你可以在llm_prompt模板.txt中看到他


### 2025-05-18

- 优化了长期以来transformers库4.50+版本的API变化与原始IndexTTS模型代码不兼容导致的生成报错问题


### 2025-05-16

- 新增对**IndexTTS-1.5**模型的支持
  - 现在可以在UI中通过下拉菜单切换不同版本的模型
  - 支持原始的Index-TTS和新的IndexTTS-1.5模型
  - 切换模型时会自动加载相应版本，无需重启ComfyUI
 
  ![微信截图_20250516182957](https://github.com/user-attachments/assets/ce13f02c-9834-43b8-82e9-5567bb226280)
  

### 2025-05-11
- 增加了seed功能，现在linux也可以重复执行抽卡了
- 增加了对 Apple Silicon MPS 设备的检测（仍需测试反馈~）


### 2025-04-23

![微信截图_20250423175608](https://github.com/user-attachments/assets/f2b15d8a-3453-4c88-b609-167b372aab74)


- 新增 **Audio Cleaner** 节点，用于处理TTS输出音频中的混响和杂音问题
  - 该节点可以连接在 Index TTS 节点之后，优化生成音频的质量
  - 主要功能：去除混响、降噪、频率滤波和音频归一化
  - 适用于处理有杂音或混响问题的TTS输出

- 修复了对于transformers版本强依赖的问题

#### Audio Cleaner 参数说明

**必需参数**：：
- **audio**: 输入音频（通常为 Index TTS 节点的输出）
- **denoise_strength**: 降噪强度（0.1-1.0，默认0.5）
  - 值越大，降噪效果越强，但可能影响语音自然度
- **dereverb_strength**: 去混响强度（0.0-1.0，默认0.7）
  - 值越大，去混响效果越强，适合处理在回声环境下录制的参考音频

**可选参数**：：
- **high_pass_freq**: 高通滤波器频率（20-500Hz，默认100Hz）
  - 用于过滤低频噪音，如环境嗡嗡声
- **low_pass_freq**: 低通滤波器频率（1000-16000Hz，默认8000Hz）
  - 用于过滤高频噪音
- **normalize**: 是否归一化音频（"true"或"false"，默认"true"）
  - 开启可使音量更均衡

#### 使用建议

- 对于有明显混响的音频，将 `dereverb_strength` 设置为 0.7-0.9
- 对于有背景噪音的音频，将 `denoise_strength` 设置为 0.5-0.8
- 如果处理后音频听起来不自然，尝试减小 `dereverb_strength` 和 `denoise_strength`
- 高通和低通滤波器可以微调以获得最佳人声效果


### 2025-04-25
- 优化了阿拉伯数字的发音判断问题；可以参考这个case使用：“4 0 9 0”会发音四零九零，“4090”会发音四千零九十； 


### 2025-04-26
- 优化英文逗号导致吞字的问题；


### 2025-04-29
- 修正了语言模式切换en的时候4090依然读中文的问题，auto现在会按照中英文占比确定阿拉伯数字读法
- 新增了从列表读取音频的方法，同时新增了一些音色音频供大家玩耍；你可以将自己喜欢的音频放入 ComfyUI-Index-TTS\TimbreModel 里，当然也很鼓励你能把好玩的声音分享出来。
- 示例用法如图：

![微信截图_20250429112255](https://github.com/user-attachments/assets/a0af9a5b-7609-4c34-adf5-e14321b379a7)


## 安装

### 安装节点

1. 将此代码库克隆或下载到ComfyUI的`custom_nodes`目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/chenpipi0807/ComfyUI-Index-TTS.git
   ```

2. 安装依赖： 安装依赖：

   ```bash
   cd ComfyUI-Index-TTS
   .\python_embeded\python.exe -m pip install -r requirements.txt

   git pull # 更新很频繁你可能需要
   ```

### 下载模型

#### 原始版本 (Index-TTS)

1. 从[Hugging Face](https://huggingface.co/IndexTeam/Index-TTS/tree/main)或者[魔搭](https://modelscope.cn/models/IndexTeam/Index-TTS)下载IndexTTS模型文件
2. 将模型文件放置在`ComfyUI/models/Index-TTS`目录中（如果目录不存在，请创建）
3. 模型文件夹结构：
   
   ```
   ComfyUI/models/Index-TTS/
   ├── .gitattributes
   ├── bigvgan_discriminator.pth
   ├── bigvgan_generator.pth
   ├── bpe.model
   ├── config.yaml
   ├── configuration.json
   ├── dvae.pth
   ├── gpt.pth
   ├── README.md
   └── unigram_12000.vocab
   ```
   
   确保所有文件都已完整下载，特别是较大的模型文件如`bigvgan_discriminator.pth`(1.6GB)和`gpt.pth`(696MB)。

#### 新版本 (IndexTTS-1.5)

1. 从[Hugging Face](https://huggingface.co/IndexTeam/IndexTTS-1.5/tree/main)下载IndexTTS-1.5模型文件
2. 将模型文件放置在`ComfyUI/models/IndexTTS-1.5`目录中（如果目录不存在，请创建）
3. 模型文件夹结构与Index-TTS基本相同，但文件大小和内容会有所不同：
   
   ```
   ComfyUI/models/IndexTTS-1.5/
   ├── .gitattributes
   ├── bigvgan_discriminator.pth
   ├── bigvgan_generator.pth
   ├── bpe.model
   ├── config.yaml
   ├── configuration.json
   ├── dvae.pth
   ├── gpt.pth
   ├── README.md
   └── unigram_12000.vocab
   ```

## 使用方法

1. 在ComfyUI中，找到并添加`Index TTS`节点
2. 连接参考音频输入（AUDIO类型）
3. 输入要转换为语音的文本
4. 调整参数（语言、语速等）
5. 运行工作流获取生成的语音输出

### 示例工作流

项目包含一个基础工作流示例，位于`workflow/workflow.json`，您可以在ComfyUI中通过导入此文件来快速开始使用。

## 参数说明

### 必需参数

- **text**: 要转换为语音的文本（支持中英文）
- **reference_audio**: 参考音频，模型会复刻其声音特征
- **model_version**: 模型版本选择，可选项：
  - `Index-TTS`: 原始模型版本（默认）
  - `IndexTTS-1.5`: 新版本模型
- **language**: 文本语言选择，可选项：
  - `auto`: 自动检测语言（默认）
  - `zh`: 强制使用中文模式
  - `en`: 强制使用英文模式
- **speed**: 语速因子（0.5~2.0，默认1.0）

### 可选参数

以下参数适用于高级用户，用于调整语音生成质量和特性：

- **temperature** (默认1.0): 控制生成随机性，较高的值增加多样性但可能降低稳定性
- **top_p** (默认0.8): 采样时考虑的概率质量，降低可获得更准确但可能不够自然的发音
- **top_k** (默认30): 采样时考虑的候选项数量
- **repetition_penalty** (默认10.0): 重复内容的惩罚系数
- **length_penalty** (默认0.0): 生成内容长度的调节因子
- **num_beams** (默认3): 束搜索的宽度，增加可提高质量但降低速度
- **max_mel_tokens** (默认600): 最大音频token数量
- **sentence_split** (默认auto): 句子拆分方式

## 音色优化建议

要提高音色相似度：

- 使用高质量的参考音频（清晰、无噪音）
- 尝试调整`temperature`参数（0.7-0.9范围内效果较好）
- 增加`repetition_penalty`（10.0-12.0）可以提高音色一致性
- 对于长文本，确保`max_mel_tokens`足够大

## 故障排除


- 如果出现“模型加载失败”，检查模型文件是否完整且放置在正确目录
- 对于Windows用户，无需额外安装特殊依赖，节点已优化
- 如果显示CUDA错误，尝试重启ComfyUI或减少`num_beams`值
- 如果你是pytorch2.7运行报错，短期无法适配，请尝试降级方案(.\python_embeded\python.exe -m pip install transformers==4.48.3)



## 鸣谢

- 基于原始[IndexTTS](https://github.com/index-tts/index-tts)模型
- 感谢ComfyUI社区的支持
- 感谢使用！
- 

## 许可证

请参考原始IndexTTS项目许可证。
