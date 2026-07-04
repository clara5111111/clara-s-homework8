import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
# 【修改点1】引入 LLM 类
from crewai.llm import LLM

# 加载环境变量
load_dotenv()

# 【修改点2】使用 crewai 自带的 LLM 类来配置 DeepSeek
# 这样就不需要导入 langchain_openai 了，兼容性最好
deepseek_llm = LLM(
    model="deepseek/deepseek-chat",   # 格式通常是 "提供商/模型名"
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# 创建 Agent
researcher = Agent(
    role='资深科技研究员',
    goal='研究并列出多智能体系统（Multi-Agent System）的3个核心优势。',
    backstory='你是一位专注于人工智能前沿技术的专家，擅长分析复杂系统。',
    verbose=True,
    allow_delegation=False,
    llm=deepseek_llm  # 【修改点3】传入上面定义好的 deepseek_llm
)

# 创建 Task
task = Task(
    description="研究并列出多智能体系统（Multi-Agent System）的3个核心优势。",
    expected_output="一份包含3个核心优势的清晰列表，每个优势需附带简要解释。",      agent=researcher
)

# 创建 Crew 并执行
crew = Crew(
    agents=[researcher],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()
print(result)