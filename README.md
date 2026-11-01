# Multi-LLM Harness Agent (Adapter Pattern)

Hệ thống Harness Agent hỗ trợ đa mô hình (Google Gemini, Anthropic Claude, và dễ dàng mở rộng thêm OpenAI, DeepSeek, v.v.) áp dụng kiến trúc phân lớp chuẩn **Adapter Pattern**.

---

## 1. Kiến trúc phân lớp & Mô hình Supervisor / Orchestrator

Hệ thống áp dụng mô hình **Supervisor (Saga Orchestrator)** điều phối tập trung:

```
                         ┌─────────────────────────────────┐
                         │       User / Task Request       │
                         └────────────────┬────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   👑 SUPERVISOR (Orchestrator)  │
                         │   - Phân tích & Lập plan        │
                         │   - Giao việc qua delegate_task │
                         │   - Đánh giá báo cáo từ Worker  │
                         └────────────────┬────────────────┘
                                          │
           ┌──────────────────────────────┼──────────────────────────────┐
           │ delegate_task                │ delegate_task                │ delegate_task
           ▼                              ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
│  🔍 Researcher Worker   │    │     💻 Coder Worker     │    │    🛡️ Reviewer Worker   │
├─────────────────────────┤    ├─────────────────────────┤    ├─────────────────────────┤
│ • Model: Flash-Lite     │    │ • Model: Flash / Claude │    │ • Model: Flash / Claude │
│ • Role: Khảo sát file   │    │ • Role: Viết & sửa code │    │ • Role: Test & kiểm tra │
│ • Tools: read_file,     │    │ • Tools: write_file,    │    │ • Tools: read_file,     │
│          list_directory │    │          run_command    │    │          run_command    │
└────────────┬────────────┘    └────────────┬────────────┘    └────────────┬────────────┘
             │ Báo cáo kết quả              │ Báo cáo kết quả              │ Báo cáo kết quả
             └──────────────────────────────┴──────────────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   👑 Phản hồi cuối cho User    │
                         └─────────────────────────────────┘
```

---

## 2. Cấu trúc thư mục

```
Harness/
├── adapters/
│   ├── base.py            # Standard Interface: Message, ToolSpec, ToolCall, BaseLLMAdapter
│   ├── gemini_adapter.py  # Google Gemini implementation
│   ├── claude_adapter.py  # Anthropic Claude implementation
│   └── factory.py         # Factory function `get_llm_adapter(model_name)`
├── tools/
│   ├── base.py            # Decorator @tool tự động convert python type hints -> JSON Schema
│   └── builtin.py         # Built-in tools: read_file, write_file, run_command, list_directory
├── agent.py               # ReAct harness loop (HarnessAgent)
├── config.py              # Load biến môi trường (.env)
├── main.py                # CLI runner (hỗ trợ one-shot prompt hoặc interactive chat)
├── .env.example           # Mẫu cấu hình API keys
└── requirements.txt       # Dependencies
```

---

## 3. Cài đặt & Sử dụng

### 3.1. Cài đặt dependencies

```bash
pip install -r requirements.txt
# hoặc dùng uv:
uv pip install -r requirements.txt
```

### 3.2. Cấu hình API Keys

Tạo file `.env` từ `.env.example`:

```bash
GEMINI_API_KEY="AIzaSy..."
ANTHROPIC_API_KEY="sk-ant-..."
DEFAULT_MODEL=gemini-2.5-flash
```

### 3.3. Chạy Agent

#### Chế độ Web UI (Khuyên dùng - Trực quan nhất):
```bash
py server.py
```
Mở trình duyệt truy cập: **`http://localhost:8000`**
- Giao diện Dark Mode trực quan, hiện luồng suy nghĩ ([THOUGHT]) và công cụ gọi ([TOOL CALL]).
- Có Dropdown chuyển đổi Model (Gemini / Claude).
- Có Sidebar hiển thị danh sách file thực tế trong Workspace.

#### Chế độ Terminal CLI (One-shot hoặc REPL):
```bash
# Chạy 1 câu lệnh
py main.py "Hãy kiểm tra xem trong thư mục hiện tại có những file gì"

# Chat trực tiếp trên terminal
py main.py --model gemini-3.1-flash-lite
```

---

## 4. Cách mở rộng thêm Model mới (ví dụ OpenAI hoặc DeepSeek)

Chỉ cần tạo thêm 1 file adapter (ví dụ: `adapters/openai_adapter.py`) kế thừa `BaseLLMAdapter`:

```python
from adapters.base import BaseLLMAdapter, LLMResponse, Message, ToolSpec

class OpenAIAdapter(BaseLLMAdapter):
    def generate(self, messages, tools=None, temperature=0.2, max_tokens=None) -> LLMResponse:
        # Convert standard messages & tools sang định dạng OpenAI
        # Gọi client.chat.completions.create(...)
        # Convert response sang LLMResponse chuẩn
        pass
```

Sau đó đăng ký vào `adapters/factory.py`. Toàn bộ core logic của Agent và Tool execution giữ nguyên 100%, không cần sửa bất kỳ dòng nào!

