from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from adapters.base import BaseLLMAdapter, ToolSpec
from adapters.factory import get_llm_adapter
from agent import HarnessAgent
from tools.builtin import BUILTIN_TOOLS


@dataclass
class AgentProfile:
    id: str
    name: str
    role_title: str
    avatar: str
    color: str  # e.g. "cyan", "purple", "emerald"
    model: str
    system_prompt: str
    allowed_tools: List[str]
    enabled: bool = True


DEFAULT_AGENTS: List[AgentProfile] = [
    AgentProfile(
        id="researcher",
        name="Researcher",
        role_title="Đọc tài liệu & Khảo sát",
        avatar="🔍",
        color="cyan",
        model="gemini-3.1-flash-lite",
        system_prompt=(
            "Bạn là Research Agent chuyên trách khảo sát môi trường và đọc tài liệu/file.\n"
            "Nhiệm vụ: Sử dụng các công cụ được cấp để đọc codebase, tìm kiếm thông tin và liệt kê các file liên quan.\n"
            "Tuyệt đối KHÔNG sửa file hay viết code thực thi.\n"
            "Kết quả trả về: Báo cáo phân tích hiện trạng và đề xuất hướng triển khai rõ ràng cho Coder Agent."
        ),
        allowed_tools=["read_file", "list_directory"],
        enabled=True,
    ),
    AgentProfile(
        id="coder",
        name="Coder",
        role_title="Lập trình & Triển khai",
        avatar="💻",
        color="purple",
        model="gemini-3.1-flash-lite",
        system_prompt=(
            "Bạn là Coder Agent chuyên trách lập trình và thực thi tác vụ.\n"
            "Nhiệm vụ: Đọc phân tích của Researcher Agent (nếu có) và yêu cầu người dùng để viết file, sửa file, hoặc chạy lệnh.\n"
            "Sử dụng công cụ write_file và run_command để triển khai giải pháp hoàn chỉnh.\n"
            "Kết quả trả về: Tóm tắt những thay đổi code đã thực hiện và kết quả chạy thử."
        ),
        allowed_tools=["read_file", "write_file", "run_command", "list_directory"],
        enabled=True,
    ),
    AgentProfile(
        id="reviewer",
        name="Reviewer",
        role_title="Kiểm duyệt & Đảm bảo chất lượng",
        avatar="🛡️",
        color="emerald",
        model="gemini-3.1-flash-lite",
        system_prompt=(
            "Bạn là Reviewer Agent (QA/Security) chuyên trách kiểm tra chất lượng sản phẩm.\n"
            "Nhiệm vụ: Đọc lại các file code mà Coder vừa tạo hoặc sửa đổi, chạy thử lệnh test (nếu cần), kiểm tra lỗi logic, bảo mật, cú pháp.\n"
            "Kết quả trả về: Đánh giá PASS/FAIL, các điểm tốt và góp ý cải thiện (nếu có)."
        ),
        allowed_tools=["read_file", "run_command", "list_directory"],
        enabled=True,
    ),
]


class TeamOrchestrator:
    """
    Điều phối nhiều Agents làm việc theo quy trình Pipeline (Saga Pattern):
    Researcher ➔ Coder ➔ Reviewer
    """

    def __init__(
        self,
        agents_config: Optional[List[AgentProfile]] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        self.profiles = agents_config or [
            AgentProfile(**vars(p)) for p in DEFAULT_AGENTS
        ]
        self.on_event = on_event
        self.tool_lookup = {t.name: t for t in BUILTIN_TOOLS}

    def emit(self, event_type: str, data: dict):
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass

    def run_pipeline(self, user_prompt: str) -> Dict[str, str]:
        active_agents = [a for a in self.profiles if a.enabled]

        if not active_agents:
            msg = "Cảnh báo: Không có Agent nào được kích hoạt trong pipeline!"
            self.emit("error", {"message": msg})
            return {"error": msg}

        self.emit("pipeline_start", {
            "total_agents": len(active_agents),
            "agents": [
                {"id": a.id, "name": a.name, "avatar": a.avatar, "color": a.color, "role_title": a.role_title}
                for a in active_agents
            ],
            "user_prompt": user_prompt,
        })

        pipeline_context = f"Yêu cầu ban đầu của người dùng:\n{user_prompt}\n"
        results: Dict[str, str] = {}

        for index, profile in enumerate(active_agents, start=1):
            self.emit("agent_start", {
                "step_index": index,
                "total_steps": len(active_agents),
                "agent_id": profile.id,
                "name": profile.name,
                "role_title": profile.role_title,
                "avatar": profile.avatar,
                "color": profile.color,
                "model": profile.model,
            })

            # Filter allowed tools for this specific agent
            agent_tools = [
                self.tool_lookup[t_name]
                for t_name in profile.allowed_tools
                if t_name in self.tool_lookup
            ]

            adapter = get_llm_adapter(profile.model)

            # Build specialized agent instance
            def forward_event(ev_type: str, data: dict):
                augmented = dict(data)
                augmented["agent_id"] = profile.id
                augmented["agent_name"] = profile.name
                augmented["agent_avatar"] = profile.avatar
                augmented["agent_color"] = profile.color
                self.emit(ev_type, augmented)

            agent = HarnessAgent(
                adapter=adapter,
                tools=agent_tools,
                system_prompt=profile.system_prompt,
                max_steps=10,
                on_event=forward_event,
                on_step_callback=lambda msg: None,
            )

            # Build input prompt for this agent stage
            if index == 1:
                stage_prompt = user_prompt
            else:
                stage_prompt = (
                    f"{pipeline_context}\n"
                    f"--- KẾT QUẢ TỪ CÁC BƯỚC TRƯỚC ---\n"
                    + "\n\n".join([f"[{k.upper()} HOÀN THÀNH]:\n{v}" for k, v in results.items()])
                    + f"\n\nNhiệm vụ của bạn ({profile.name} - {profile.role_title}): Hãy tiếp tục thực hiện công việc theo đúng chuyên môn của bạn."
                )

            agent_output = agent.run(stage_prompt)
            results[profile.id] = agent_output

            self.emit("agent_complete", {
                "agent_id": profile.id,
                "name": profile.name,
                "avatar": profile.avatar,
                "output": agent_output,
            })

        self.emit("pipeline_complete", {
            "summary": results,
            "final_agent": active_agents[-1].name,
            "final_output": results[active_agents[-1].id],
        })

        return results

