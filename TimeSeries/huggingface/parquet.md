# Hugging Face Dataset 深度指南：从 Parquet 打包到高效训练

在现代机器学习项目中，数据是模型的“燃料”。然而，如何**高效地组织、存储和加载数据**，常常成为瓶颈。Hugging Face 的 `datasets` 库提供了一套强大而优雅的解决方案，它不仅支持多种格式（如 CSV、JSON、Parquet），还通过 **Apache Arrow** 实现了高性能的数据读取与处理。

---

## 一、为什么选择 Parquet + Arrow

### ✅ 为什么用 Parquet？

在 Hugging Face 生态中，你可能已经注意到：几乎所有公开数据集（如 glue, squad, imdb）在 Hub 上都是以 .parquet 文件形式存储的。

这不是偶然——Parquet 是现代机器学习数据管道的事实标准格式。

**Parquet** 是一种高效的列式存储文件格式，特别适合以下场景：

| 场景 | 是否适合 |
|------|----------|
| 只需读取部分字段（如只用 `text` 和 `label`） | ✅ 极快（列裁剪） |
| 数据量大（GB~TB 级别） | ✅ 压缩率高，I/O 效率好 |
| 多次重复使用 | ✅ 支持缓存，下次秒开 |
| 分布式训练 | ✅ Spark/Dask/HF 都原生支持 |

**Apache Arrow** 是一个开源的内存列式数据格式标准，专为分析型工作负载设计。它的核心优势包括：
- **与 Parquet 无缝集成**：Parquet 是磁盘上的列式存储格式，Arrow 是内存中的列式格式。

> 💡 Hugging Face Datasets 底层正是基于 Arrow 构建，所有数据最终都会被转换为 Arrow Table。

> 🎯 **结论**：如果你有结构化数据（如 NLP、语音、时间序列），**Parquet 是最佳选择之一**。

---

## 二、如何将你的数据集打包成 Parquet 格式？

假设你有一个新的 NLP 任务：**中文情感分析**。你收集了 10 万条微博文本，并标注了情绪类别（正面/负面/中性）。每条记录包含：

```python
{
    "id": "00001",
    "text": "今天天气真好，心情不错！",
    "label": "positive",
    "length": 12,
    "source": "weibo"
}
```

### 步骤 1：准备原始数据

你可以从 CSV 或 JSON 开始，但最终目标是生成 Parquet 文件。

```python
import pandas as pd
from datasets import Dataset, DatasetDict

# 示例：从 DataFrame 转换
df = pd.read_csv("raw_data.csv")
dataset = Dataset.from_pandas(df)
```

### 步骤 2：按逻辑分区保存（推荐）

为了便于管理和后续加载，我们的文件保存结构如下：

```
sentiment_dataset/
├── train/
│   ├── part-00000.parquet
│   └── ...
├── validation/
│   ├── part-00000.parquet
│   └── ...
└── test/
│   ├── part-00000.parquet
│   └── ...
```

```python
import os
import pandas as pd
from datasets import Dataset

# 从原始文件加载（例如 CSV）
df = pd.read_csv("raw_data.csv")

# 转为 Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# 划分：80% train, 10% validation, 10% test
train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = {
    "train": train_testvalid["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"]
}


output_base = "./sentiment_dataset"
for split_name, ds in final_dataset.items():
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # 保存为单个 Parquet 文件（也可循环分片写入多个）
    parquet_path = os.path.join(split_dir, "data.parquet")
    ds.to_parquet(parquet_path)
    print(f"Saved {len(ds)} samples to {parquet_path}")
```

> ✅ 这种结构清晰对应训练流程，且与 Hugging Face Hub 上大多数数据集的组织方式一致（如 `imdb`, `glue/mnli` 等）。
>
> 🔧 如果未来数据量变大，你可以在 `to_parquet` 前对 `ds` 进行分块（例如每 5 万样本一个文件），生成 `part-00000.parquet`, `part-00001.parquet` 等，但核心逻辑不变。

---

## 三、如何加载 Parquet 数据集？三种方式详解

当你完成打包后，就可以轻松加载了。以下是三种常用方法：

`我们以train_dataset来举例，对于valid也是一样的`

### 方法 1：使用通配符（最推荐，最简单）

这种方式可以自动递归搜索目录下所有的 Parquet 文件。

```python
from datasets import load_dataset

# 使用 ** 匹配所有子文件夹中的所有 .parquet 文件
dataset = load_dataset(
    "parquet",
    data_files="./sentiment_dataset/train/*.parquet",
    split="train"  # 指定返回的是训练集
)

print(dataset)
```


### 方法 2：流式加载（针对显存/内存不足的超大数据集）

如果总数据量超过你的内存容量（如 130GB+），千万不要直接加载，应使用 `streaming=True`。

```python
dataset = load_dataset(
    "parquet",
    data_files="./sentiment_dataset/train/*.parquet",
    split="train",
    streaming=True  # 开启流式读取，不占用磁盘/内存空间下载或缓存
)

# 像迭代器一样使用
for example in dataset:
    print(example["text"])
    break
```

> ⚠️ 注意：流式加载无法进行全局打乱（Shuffle），只能在缓冲区内打乱。

---

## 四、进阶：如何保持数据打乱（Shuffle）？

在多个 Parquet 文件一起训练时，为了保证模型不按类别“死记硬背”，你需要进行打乱。

### 如果是常规加载：

```python
dataset = dataset.shuffle(seed=42)
```

### 如果是流式加载（Streaming）：

流式加载无法进行全局打乱，它会维护一个缓冲区（Buffer）：

```python
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10000)
```

> ✅ `buffer_size` 越大，打乱效果越好，但占用内存越多。

---

## 五、预处理：两种方式对比

即使数据已经保存为 Parquet，你也可能需要在训练前做预处理（如分词、标准化、插值等）。

### 方法 A：逐样本处理（`batched=False`）

```python
def preprocess(example):
    # 对单个样本做处理
    tokens = tokenizer(example["text"], truncation=True, padding="max_length")
    return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

dataset_processed = dataset.map(preprocess, num_proc=8)
```

> ❌ 缺点：函数调用次数多，性能低。

### 方法 B：批量处理（`batched=True`）✅ 推荐！

```python
def preprocess_batch(batch):
    texts = batch["text"]
    inputs = tokenizer(texts, truncation=True, padding="max_length", return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"].tolist(),
        "attention_mask": inputs["attention_mask"].tolist()
    }

dataset_processed = dataset.map(
    preprocess_batch,
    batched=True,
    batch_size=1000,
    num_proc=8
)
```

> ✅ 优势： 函数调用次数减少，100 万个样本 → 只调用函数 1000 次

#### 问题：  
> **“做训练的时，batch size 是否需要和 map 里的 batch 大小一致？”**

#### 📌 答案：**完全不需要！两者毫无关系，可以任意设置。**

> 💡 类比：  
> `.map(batch_size)` 像是“厨师一次炒 1000 份菜备用”；  
> `DataLoader(batch_size)` 像是“服务员每次端 32 份上桌”。  
> 厨师炒多少，不影响服务员端多少。

---

## 六、如何将 Dataset 用于训练？两种方式

### 方式 1：传统 PyTorch + DataLoader

```python
from torch.utils.data import DataLoader

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 在训练循环中使用
for batch in dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    # ... 模型前向传播
```

### 方式 2：Hugging Face Trainer（推荐）

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    evaluation_strategy="epoch",  # 每个 epoch 后评估
    logging_dir="./logs",
    do_eval=True,                 # 启用验证
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],  # 显式传入验证集
    tokenizer=tokenizer,
)

trainer.train()
```

> ✅ Trainer 会自动处理 `Dataset` 的 `__getitem__` 和 `__len__`，支持流式加载、混合精度等高级功能。

---

## 七、为什么这样做很快？

Hugging Face 的后端是 **Apache Arrow**。当你调用 `load_dataset` 时，它并不会把 130GB 数据全部塞进 RAM，而是利用 **内存映射（Memory Mapping）**。

它只在你的模型真正需要某一行数据时，才从磁盘读取对应的分片。

> 💡 建议：如果你的机器内存（RAM）小于 128GB，强烈建议使用 **流式加载（Streaming）** 进行训练。

---

## ✅ 总结：构建高效数据管道的最佳实践

| 步骤 | 推荐做法 |
|------|----------|
| **数据整理** | 使用 Pandas 或 Dataset 从原始数据构建结构化数据 |
| **存储格式** | ✅ 优先选择 **Parquet + Arrow** |
| **分区策略** | 按标签、时间等逻辑分区，便于后续筛选 |
| **文件大小** | 单个 Parquet 文件控制在 **200–500 MB** |
| **加载方式** | 小数据集用常规加载，大数据集用 `streaming=True` |
| **预处理** | ✅ 使用 `batched=True` 提升效率 |
| **训练接入** | 推荐使用 `Trainer`，也可用 `DataLoader` |

