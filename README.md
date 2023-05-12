## ElasticChatGLM 介绍

ElasticChatGLM 是一个开源的、支持知识动态扩展的对话语言模型，[chatglm](https://github.com/THUDM/ChatGLM-6B) 基础上进行构建。ElasticChatGLM 使用了和 langchain 相似的技术，针对对话进行了优化。 

## 使用方式

### 硬件需求

| **量化等级**   | **最低 GPU 显存**（推理） | **最低 GPU 显存**（高效参数微调） |
| -------------- | ------------------------- | --------------------------------- |
| FP16（无量化） | 13 GB                     | 14 GB                             |
| INT8           | 8 GB                     | 9 GB                             |
| INT4           | 6 GB                      | 7 GB                              |

### 环境安装
除了需要满足 [chatglm](https://github.com/THUDM/ChatGLM-6B) 的安装环境外，还需要部署 [ElasticFlow](https://github.com/springwings/elasticflow) 与ElasticSearch搜索引擎。
ElasticSearch安装不再赘述。
### 部署步骤 
* 1) 下载ElasticFlow 5.6.1 jar包，并参考 [elasticflow部署运行](https://github.com/springwings/elasticflow/wiki/v5.x-%E9%83%A8%E7%BD%B2%E8%BF%90%E8%A1%8C) 进行部署。  
* 2) 拷贝本项目下的 config.properties 到 /opt/EF/config 文件夹下;
* 3) 拷贝本项目下的 resource.xml 到 /opt/EF/datas 文件夹下，注意修改csv_folder该项的地址以及kges中的elasticsearch的地址;
* 4) 拷贝本项目下的 files_es 文件夹到 /opt/EF/datas/INSTANCES 文件夹下;
* 4) 拷贝本项目下的 kg 知识数据文件夹到 /opt/EF/ 文件夹下;
* 5) 启动ElasticFlow
* 5) 修改web_demo2.py中的ef_path变量为ElasticFlow主机地址，启动web_demo2.py。

### 说明 
知识更新参照 kg 知识数据文件夹中的范例往下添加即可，系统会在线自动更新系统整体知识，注意其中 id字段不能重复,content字段为知识的主体内容
