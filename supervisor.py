import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from adapters.base import BaseLLMAdapter, ToolSpec
from adapters.factory import get_llm_adapter
from agent import HarnessAgent
from tools.base import tool
from tools.builtin import BUILTIN_TOOLS


@dataclass
class WorkerProfile:
    id: str
    name: str
    role_title: str
    avatar: str
    color: str
    model: str
    system_prompt: str
    allowed_tools: List[str]
    enabled: bool = True


DEFAULT_WORKERS: List[WorkerProfile] = [
    WorkerProfile(
        id="researcher",
        name="Researcher",
        role_title="Khảo sát & Đọc tài liệu",
        avatar="🔍",
        color="cyan",
        model="gemini-3.1-flash-lite",
        system_prompt=(
            "Bạn là Researcher Worker chuyên trách khảo sát hệ thống, đọc file và tìm kiếm thông tin.\n"
            "Chỉ sử dụng công cụ đọc (`read_file`, `list_directory`). Tuyệt đối không tạo/sửa file.\n"
            "Báo cáo chi tiết kết quả cho Supervisor."
        ),
        allowed_tools=["read_file", "list_directory"],
        enabled=True,
    ),
    WorkerProfile(
        id="coder",
        name="Coder",
        role_title="Lập trình & Thực thi",
        avatar="💻",
        color="purple",
        model="gemini-3.1-flash-lite",
        system_prompt=(
            "Bạn là Coder Worker chuyên trách lập trình, tạo/sửa file và chạy lệnh.\n"
            "Thực hiện chính xác nhiệm vụ được Supervisor giao bằng các công cụ `write_file`, `run_command`, `read_file`.\n"
            "Báo cáo kết quả công việc và tóm tắt những file đã tạo/sửa cho Supervisor."
        ),
        allowed_tools=["read_file", "write_file", "run_command", "list_directory"],
        enabled=True,
    ),
    WorkerProfile(
        id="reviewer",
        name="Reviewer",
        role_title="Kiểm duyệt QA & Test",
        avatar="🛡️",
        color="emerald",
        model="gemini-3.1-flash-lite",
        system_prompt=(
            "Bạn là Reviewer Worker chuyên trách thẩm định chất lượng, an toàn và chạy test kiểm tra code.\n"
            "Đọc lại các file code được giao (`read_file`) hoặc chạy lệnh test (`run_command`).\n"
            "Báo cáo cho Supervisor kết luận ĐẠT (PASS) hoặc KHÔNG ĐẠT (FAIL kèm lý do lỗi chi tiết)."
        ),
        allowed_tools=["read_file", "run_command", "list_directory"],
        enabled=True,
    ),
]


SUPERVISOR_SYSTEM_PROMPT = """Bạn là Supervisor (Quản lý trưởng / Orchestrator) chịu trách nhiệm điều phối toàn bộ hệ thống Multi-Agent.
Mô hình hoạt động (Saga Orchestrator):
1. Người dùng gửi yêu cầu -> Bạn phân tích bài toán, lên kế hoạch từng bước.
2. Bạn KHÔNG tự làm việc thực thi trực tiếp, mà SỬ DỤNG TOOL `delegate_task` để giao việc cho các Worker cấp dưới:
   - "researcher": Khảo sát thư mục, đọc tài liệu, tìm hiểu mã nguồn.
   - "coder": Viết mã, tạo file, sửa code hoặc chạy lệnh thực thi.
   - "reviewer": Soát lỗi code vừa viết, chạy test, đánh giá chất lượng.
3. Khi nhận báo cáo từ Worker:
   - Bạn đánh giá kết quả: Nếu reviewer báo lỗi, bạn có thể giao lại cho Coder sửa tiếp!
   - Nếu mọi việc đã hoàn thành hoặc không cần gọi worker nào nữa, bạn tổng hợp câu trả lời cuối cùng cho người dùng.

LƯU Ý QUAN TRỌNG:
- Mỗi lần gọi `delegate_task`, hãy chỉ rõ `worker_id` và mô tả nhiệm vụ thật cụ thể.
- Chỉ giao việc cho các worker đang có trong danh sách được kích hoạt.
"""


class SupervisorOrchestrator:
    """
    Điều phối tập trung (Centralized Supervisor Orchestrator):
    Supervisor (Manager) ──delegate_task──► Worker (Researcher / Coder / Reviewer)
    Worker ──báo cáo kết quả──► Supervisor
    Supervisor quyết định bước tiếp theo hoặc kết luận.
    """

    def __init__(
        self,
        supervisor_model: str = "gemini-3.1-flash-lite",
        workers_config: Optional[List[WorkerProfile]] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        self.supervisor_model = supervisor_model
        self.workers = workers_config or [WorkerProfile(**vars(w)) for w in DEFAULT_WORKERS]
        self.on_event = on_event
        self.tool_lookup = {t.name: t for t in BUILTIN_TOOLS}
        self.worker_map = {w.id: w for w in self.workers}

    def emit(self, event_type: str, data: dict):
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass

    def _execute_worker_task(self, worker_id: str, instructions: str) -> str:
        worker_profile = self.worker_map.get(worker_id)
        if not worker_profile:
            return f"Lỗi: Không tìm thấy worker có id '{worker_id}'."

        if not worker_profile.enabled:
            return f"Lỗi: Worker '{worker_profile.name}' hiện đang bị TẮT bởi người dùng. Vui lòng chọn worker khác hoặc tự xử lý."

        self.emit("supervisor_delegates", {
            "worker_id": worker_profile.id,
            "worker_name": worker_profile.name,
            "worker_avatar": worker_profile.avatar,
            "worker_color": worker_profile.color,
            "instructions": instructions,
        })

        # Build tools for worker
        worker_tools = [
            self.tool_lookup[t_name]
            for t_name in worker_profile.allowed_tools
            if t_name in self.tool_lookup
        ]

        # Forward worker sub-events with worker tag
        def forward_worker_event(ev_type: str, data: dict):
            augmented = dict(data)
            augmented["worker_id"] = worker_profile.id
            augmented["worker_name"] = worker_profile.name
            augmented["worker_avatar"] = worker_profile.avatar
            augmented["worker_color"] = worker_profile.color
            self.emit(f"worker_{ev_type}", augmented)

        worker_adapter = get_llm_adapter(worker_profile.model)
        worker_agent = HarnessAgent(
            adapter=worker_adapter,
            tools=worker_tools,
            system_prompt=worker_profile.system_prompt,
            max_steps=10,
            on_event=forward_worker_event,
            on_step_callback=lambda msg: None,
        )

        worker_report = worker_agent.run(instructions)

        self.emit("worker_reports", {
            "worker_id": worker_profile.id,
            "worker_name": worker_profile.name,
            "worker_avatar": worker_profile.avatar,
            "worker_color": worker_profile.color,
            "report": worker_report,
        })

        return f"BÁO CÁO TỪ {worker_profile.name.upper()}:\n{worker_report}"

    def run(self, user_prompt: str) -> str:
        active_workers = [w for w in self.workers if w.enabled]
        active_workers_desc = "\n".join([
            f"- id: '{w.id}' | Tên: {w.name} | Vai trò: {w.role_title} | Model: {w.model}"
            for w in active_workers
        ])

        system_prompt = (
            f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
            f"DANH SÁCH WORKER HIỆN CÓ SẴN (ĐÃ BẬT):\n{active_workers_desc}\n"
        )

        self.emit("supervisor_start", {
            "supervisor_model": self.supervisor_model,
            "active_workers": [
                {"id": w.id, "name": w.name, "avatar": w.avatar, "color": w.color, "role": w.role_title}
                for w in active_workers
            ],
            "user_prompt": user_prompt,
        })

        # Define the delegation tool for Supervisor
        delegate_tool = ToolSpec(
            name="delegate_task",
            description=(
                "Giao một nhiệm vụ cụ thể cho một Worker chuyên trách (researcher, coder, hoặc reviewer). "
                "Worker sẽ thực thi và trả về báo cáo kết quả."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "worker_id": {
                        "type": "string",
                        "description": "ID của worker cần giao việc: 'researcher', 'coder', hoặc 'reviewer'.",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Mô tả chi tiết nhiệm vụ và dữ liệu cần xử lý gửi cho worker.",
                    },
                },
                "required": ["worker_id", "instructions"],
            },
            handler=self._execute_worker_task,
        )

        # Supervisor events forwarding
        def forward_supervisor_event(ev_type: str, data: dict):
            augmented = dict(data)
            augmented["is_supervisor"] = True
            self.emit(f"supervisor_{ev_type}", augmented)

        supervisor_adapter = get_llm_adapter(self.supervisor_model)
        supervisor_agent = HarnessAgent(
            adapter=supervisor_adapter,
            tools=[delegate_tool],
            system_prompt=system_prompt,
            max_steps=12,
            on_event=forward_supervisor_event,
            on_step_callback=lambda msg: None,
        )

        final_response = supervisor_agent.run(user_prompt)

        self.emit("supervisor_complete", {
            "final_response": final_response,
        })

        return final_response

