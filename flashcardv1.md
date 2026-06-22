Flashcard v1 — sinh theo chủ đề, ôn và lưu tiến độ
Tóm tắt
Tích hợp flashcard vào kiến trúc chính backend/ + frontend/: người dùng nhập chủ đề, chọn 5/10/15 thẻ, hệ thống truy xuất các chunk liên quan trong session rồi sinh bộ thẻ mới. Sinh lại sẽ thay thế toàn bộ bộ thẻ cũ. Mỗi thẻ có chế độ lật và trạng thái not_reviewed, learning, known, được lưu theo session.
Thay đổi chính
Thêm migration và model Flashcard trong Postgres: session_id, câu hỏi, đáp án, trạng thái ôn, tên tài liệu/trang nguồn, thời điểm tạo/cập nhật. Không phụ thuộc khóa ngoại vào chunk để thẻ vẫn tồn tại khi tài liệu được ingest lại.
Thêm API trong backend:POST /api/v1/sessions/{session_id}/flashcards/generate nhận topic và count (5 | 10 | 15); truy xuất chunk của session theo chủ đề, gọi Gemini theo structured JSON, kiểm tra nguồn trả về hợp lệ, xóa bộ thẻ cũ rồi lưu bộ mới.
GET /api/v1/sessions/{session_id}/flashcards trả các thẻ và thống kê theo trạng thái.
PATCH /api/v1/sessions/{session_id}/flashcards/{flashcard_id} cập nhật trạng thái ôn.

Bổ sung prompt/generator flashcard riêng trong luồng RAG chính: câu hỏi–đáp án bằng tiếng Việt, ngắn gọn, chỉ dựa trên context; mỗi thẻ chỉ rõ chỉ mục nguồn để backend gắn tên PDF/trang tương ứng.
Mở rộng frontend API client/types và thay panel placeholder bằng trải nghiệm thật:Ô nhập chủ đề, chọn 5/10/15 thẻ, nút Generate.
Trạng thái loading/error; disable khi session chưa có tài liệu ready.
Hiển thị từng thẻ, chạm/click để lật xem đáp án.
Sau khi lật, cho chọn Not reviewed, Learning, Known; đồng bộ ngay với backend.
Hiển thị số lượng theo từng trạng thái, progress bar và nguồn rút gọn trên mặt đáp án.
Sinh lại hiển thị cảnh báo rõ rằng bộ thẻ hiện tại sẽ bị thay thế.

Kiểm thử
Migration tạo/xóa bảng và ràng buộc dữ liệu hợp lệ.
Unit test generator: structured output hợp lệ, nguồn không hợp lệ bị từ chối, thiếu context/Gemini lỗi trả lỗi rõ ràng.
API test: chỉ session tồn tại mới được sinh/lấy/cập nhật thẻ; generate thay thế bộ cũ; chỉ chấp nhận count 5/10/15; cập nhật trạng thái hợp lệ.
UI: render empty/loading/error, tạo thẻ theo topic, lật thẻ, cập nhật trạng thái và thống kê.
Chạy pytest, ruff check, frontend lint và build.
Giả định
V1 không có spaced repetition hay lịch ôn; đó là bước sau.
Flashcard chỉ thuộc session hiện tại và sinh từ topic do người dùng nhập.
Chưa tích hợp PDF viewer; nguồn chỉ hiển thị tên tài liệu và số trang.
Các endpoint src/ tách rời không được dùng hay mở rộng cho tính năng này.