from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables import RunnablePassthrough

import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
knowledge_dir = os.path.join(curr_dir, "knowledge")

class DeepSeek:
    """
        该类封装了集成RAG技术和历史消息的大语言模型
        使用ask_llm即可以与大语言模型进行对话
        但是，在使用ask_llm前必须使用build_chain方法，构建langchain链
        chang_location方法用于更改机器人所处的位置，提供更准确的路线指引
    """
    # 类构造函数
    def __init__(self):
        # 机器人所处的位置
        self.location = "红门"
        # Ollama本地端口以及向量数据库位置
        self.base_url = "http://localhost:11434"
        self.knowledge_base_dir = os.path.join(knowledge_dir, "chroma_db")
        # 创建Ollama大语言模型模型与Ollama嵌入模型
        self.llm = ChatOllama(model="thirdeyeai/DeepSeek-R1-Distill-Qwen-7B-uncensored:Q4_0", base_url=self.base_url)
        self.embedding = OllamaEmbeddings(model="bge-m3:latest", base_url=self.base_url)
        # 存储历史信息的字典
        self.store = {}
        # 用于大语言模型的回答
        self.result = None
        # langchain链路
        self.chain = None

        # 创建Chroma向量数据库
        self.vector = Chroma(embedding_function=self.embedding, persist_directory=self.knowledge_base_dir)
        # 创建向量检索器
        self.retriever = self.vector.as_retriever()

    # 聊天历史检索函数
    def _get_session_history(self,user_id) -> BaseChatMessageHistory:
        if user_id not in self.store:
            self.store[user_id] = ChatMessageHistory()
        return self.store[user_id]

    # 向量数据库检索函数
    def _retrieve_docs(self,inputs):
        query = inputs["query"]
        docs = self.retriever.invoke(query)
        docs_str = "\n\n".join(doc.page_content for doc in docs)
        inputs["context"] = docs_str

        return inputs

    # 构建langchain链路
    def build_langchain(self):
        # 创建输出器
        parser = StrOutputParser()
        # 创建提示词模板
        system_template =  """   你叫泰泰，是泰山景区的景区安防向导机器人，你的任务是根据下述给定的已知信息回答用户的问题。
                                    游客数量:{num}                       
                                    游客的性别年龄分别为{agegender}
                                    已知信息:{context}
                                    你当前所处的位置:{location}
                                    从红门到玉皇顶的路线如下，如果用户询问请按如下告诉他
                                    红门：泰山经典登山起点，标志性的红门建筑古色古香，充满仪式感，是必打卡的拍照点。可以在此拍照留念，开启登山之旅。
                                    万仙楼：这里有隐真洞和 128 位形态各异的神仙塑像，盘山道西侧石壁上还有 “风月无边” 石刻，极具文化底蕴，可在此欣赏古建筑和石刻，感受历史韵味。
                                    经石峪：有中国现存规模最大的佛经摩崖刻石，在面积 2064 平方米的缓坡石坪上刻有隶书《金刚经》，字体雄浑古朴，与自然景观完美融合，是拍照打卡的好地方，可在此拍出文艺范十足的照片。
                                    中天门：泰山的半山腰，是重要的休息补给点和交通枢纽。在这里可以稍作休息，补充能量，遥望南天门，还能购买纪念品，以中天门的建筑和远处的山峦为背景拍照，也很出片。
                                    十八盘：泰山最险峻的路段，1600 多级台阶，两侧山峰陡峭，山谷幽深。攀登过程虽然具有挑战性，但征服它会有很大的成就感，沿途可以拍摄自己攀登的身影以及壮观的山势，记录下这一难忘的经历。
                                    南天门：泰山的标志性建筑，云雾缭绕时宛如仙境。以其为背景拍照，壮阔的山峦能让照片极具视觉冲击力，仿佛踏入仙境，是必打卡的网红地点。
                                    天街：山顶的 “小街市”，店铺林立，古建筑与远处的山峦相互映衬。在这里可以边逛边吃，购买特色小吃和纪念品，以天街为背景拍照，十分有意境，能体现出山顶独特的氛围。
                                    碧霞祠：泰山上最大的道教建筑群，香火旺盛，红墙碧瓦与青山相映，是很好的拍照背景。可以在此祈福，感受道教文化，拍摄具有古典美的照片。
                                    唐摩崖：有唐玄宗李隆基御书的《纪泰山铭》，字体雄浑大气，展现出深厚的历史文化底蕴。在此拍照可以感受历史的厚重，让照片富有文化内涵。
                                    五岳独尊石：泰山的标志性景点之一，“五岳独尊” 四个大字苍劲有力，几乎每位游客都会在此合影留念。记得带上 5 元纸币，来一张与五岳独尊石的创意合影，称霸朋友圈。
                                    玉皇顶：泰山主峰，海拔 1545 米，站在山顶可俯瞰整个泰山的壮丽景色。以玉皇顶石碑和连绵山脉为背景拍照，能拍出气势磅礴的感觉。此外，这里还是观日出的绝佳地点，如果时间合适，还可以欣赏到美丽的日出景色。
                                    拱北石：位于泰山顶峰北侧，又称探海石，在此可俯瞰群山连绵、云海翻腾，仿佛大海波涛，拍照效果极佳，是一个能拍出壮美风光的网红打卡地。
                                    
                                    请用中文回答用户的问题。
                                    """
        chat_template = ChatPromptTemplate.from_messages([
                                                                ("system", system_template),
                                                                MessagesPlaceholder(variable_name="history"),
                                                                ("human", "{query}"),
                                                            ])
        # 创建langchain链路(不带历史消息)
        runnable = (
                RunnablePassthrough.assign(context=self._retrieve_docs)
                | chat_template
                | self.llm
                | parser
        )

        # 为langchain链路添加历史消息
        with_message_history = RunnableWithMessageHistory(
            runnable,
            self._get_session_history,
            input_messages_key="query",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(id="user_id",
                                      annotation=str,
                                      name="User ID",
                                      description="用户的唯一标识符。",
                                      default="",
                                      is_shared=True, ),
            ],
        )

        self.chain = with_message_history

    # 向大语言模型提问
    def Ask_LLM(self,num:int,agegender:str,query:str,user_id:str):
        try:
            result = ""
            for chunk in self.chain.stream(
                    {"num": num, "agegender": agegender, "query": query, "location":self.location},
                    config={"configurable": {"user_id": user_id}},
            ):
                print(chunk, end="", flush=True)
                result += chunk
            print("\n")
            return result
        except Ellipsis as e:
            print("出现错误: e")
            print("请检查Ollama服务是否正常启动")

    # 改变位置
    def change_location(self,location:str):
        self.location = location

