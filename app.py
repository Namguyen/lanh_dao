from core import process_query
from es import AICandidateDB


def main():
    try:
        db = AICandidateDB()
        print("Trạng thái: Đã kết nối cơ sở dữ liệu nội bộ.")
    except Exception as exc:
        print(f"[Lỗi] Kết nối DB thất bại: {exc}")
        return

    print("Hệ thống tra cứu sẵn sàng. Nhập 'exit' để thoát.\n")

    while True:
        try:
            user_input = input("\nCâu hỏi của bạn: ")
            if user_input.lower() in ("exit", "quit", "clear"):
                break
            if not user_input.strip():
                continue

            result = process_query(user_input, db)
            print("\n" + result["answer"])

        except KeyboardInterrupt:
            print("\nĐang đóng hệ thống...")
            break
        except Exception as exc:
            print(f"[Lỗi]: {exc}")


if __name__ == "__main__":
    main()
