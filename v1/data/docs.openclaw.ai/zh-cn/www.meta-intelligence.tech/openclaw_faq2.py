# https://www.meta-intelligence.tech/insight-openclaw-faq
qa_data = [
    {
        "query": """OpenClaw 要錢嗎？""",
        "pos": """OpenClaw 軟體本身完全免費且開源。但它需要連接 AI 模型來「思考」，而這些模型的 API 呼叫需要付費。以 Claude Sonnet 為例，一般日常使用的月費大約在 $5–$30 美元之間，取決於使用頻率。""",
    },
    {
        "query": """我的電腦跑得動嗎？""",
        "pos": """OpenClaw 對硬體的要求很低——因為「思考」的工作由雲端 AI 模型完成，你的電腦只負責執行指令。基本需求：[6]

作業系統：macOS、Linux（含 Windows WSL2）
Node.js：22 或以上
記憶體：2GB 以上（推薦 4GB）
磁碟空間：約 500MB
網路：需要穩定的網際網路連線（用於呼叫 AI 模型 API）
甚至一台樹莓派（Raspberry Pi）也能運行 OpenClaw。"""
    },
    {
        "query": """安全嗎？我的資料會被看到嗎？""",
        "pos": """安全性取決於你的配置。[5] 需要注意兩件事：

代理權限：OpenClaw 以你的用戶身份運行，能存取你能存取的所有檔案。使用 Docker 沙盒可以限制其存取範圍
資料傳輸：代理會將你的指令和相關檔案內容發送到 AI 模型供應商（如 Anthropic）進行處理。敏感資料不應讓代理接觸
詳細的安全指南請參閱我們的《安全性完全指南》。"""
    },
    {
        "query": """需要會寫程式嗎？""",
        "pos": """不需要。基本的安裝與使用只需要在終端機中輸入幾條指令（像是 npm install），不需要任何程式設計知識。之後的所有操作都是用自然語言——你用中文告訴代理要做什麼就好。"""
    },
    {
        "query": """OpenClaw 能用中文嗎？""",
        "pos": """完全可以。OpenClaw 本身只是一個代理框架，語言能力來自底層的 AI 模型。主流模型（Claude、GPT-4、Gemini）都支援中文對話與指令。你可以用中文下達任何指令，代理也會用中文回覆。"""
    },
    {
        "query": """跟 Manus AI 有什麼不同？""",
        "pos": """Manus AI 是一個商業化的 AI 代理產品，運行在雲端。OpenClaw 是開源的，運行在你自己的電腦上。主要差異：

控制權：OpenClaw 的代理直接操作你的電腦，Manus 在雲端沙盒中運行
開源 vs 商業：OpenClaw 完全開源可自由修改，Manus 是封閉的商業服務
費用模型：OpenClaw 按 API 使用量付費，Manus 按訂閱制收費
更詳細的比較請參閱《OpenClaw vs Manus 深度比較》。"""
    },
    {
        "query": """手機上能用嗎？""",
        "pos": """OpenClaw 必須安裝在一台電腦（或伺服器）上運行。但你可以透過 Telegram、WhatsApp 等手機 App 來遠端控制它。效果等同於用手機操控你的電腦。"""
    },
    {
        "query": """OpenClaw 的名稱由來？""",
        "pos": """OpenClaw 最初名為 ClawdBot，後改名 MoltBot，最終定名 OpenClaw。這段名稱演變反映了社群主導的開源發展過程。"""
    },
    {
        "query": """
""",
        "pos": """
"""
    },
 
]
