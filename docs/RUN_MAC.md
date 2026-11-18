# Chạy HVI-CIDNet trên Mac Apple Silicon

Môi trường riêng: `.venv`, Python 3.11. Bộ thư viện tương thích nằm trong
`requirements-macos.txt`; `requirements.txt` của tác giả được giữ nguyên.

## Mở giao diện

Trong Terminal, chạy:

```bash
cd "/Users/huynguyen/Data/Master/10. Multimedia/Project/HVI-CIDNet"
source .venv/bin/activate
python -m upstream.app --cpu
```

Mở http://127.0.0.1:7862, chọn ảnh thiếu sáng, giữ `Image Score = No`,
chọn checkpoint `LOLv2_real/best_PSNR.pth`, giữ gamma/Alpha-s/Alpha-i bằng 1
và bấm Submit. Ảnh thử: `examples/low/1.png`.

`Processing size (long edge)` mặc định **768**: ảnh lớn được thu nhỏ, giữ đúng
tỷ lệ, để thử nhanh trên CPU. Kích thước đầu vào/đầu ra và thời gian xuất hiện
trong `Processing details`. Ảnh nhỏ hơn giới hạn không bị phóng lớn.
Chọn **Original** để giữ nguyên độ phân giải; ảnh điện thoại 24 MP có thể chạy
rất lâu trên CPU. Khi đo chỉ số cho paper, dùng cùng một quy tắc độ phân giải
cho tất cả phương pháp và ảnh tham chiếu; không so trực tiếp kết quả thu nhỏ
với ảnh đáp án kích thước gốc. `eval_hf.py` tiếp tục giữ nguyên độ phân giải.

Có thể nhấp đúp `scripts/run_demo.command` trong Finder để chạy bằng CPU.
Dừng bằng Ctrl+C trong cửa sổ Terminal.

## Chạy một ảnh và lưu kết quả

```bash
python -m upstream.eval_hf --device cpu \
  --path Fediory/HVI-CIDNet-LOLv2-real-bestPSNR \
  --input_img examples/low/1.png
```

Đầu ra lưu ở `output_hf/1.png`. Lần đầu cần mạng để tải checkpoint;
những lần sau dùng cache Hugging Face. Có thể thay `--input_img` bằng đường dẫn
đến ảnh của bạn, đặt đường dẫn trong dấu ngoặc kép nếu có khoảng trắng.

## Phạm vi

- Dùng checkpoint của tác giả, chạy suy luận trên CPU; chưa huấn luyện lại.
- Chỉnh đường dẫn của demo để chạy được macOS; thêm CPU cho `eval_hf.py`.
- Khi bật `Image Score = Yes`, dùng lớp tương thích BRISQUE với API mới của
  scikit-image/SciPy; NIQE giữ hàm của repo. Các điểm này không thay thế đánh giá
  bằng ảnh tham chiếu.
- Không thay đổi kiến trúc hoặc thuật toán tăng cường ảnh.
- Giữ tham số cố định khi so sánh ảnh; không điều chỉnh đầu ra theo ảnh đáp án.
- Ảnh mẫu lấy từ repo RetinexNet:
  https://github.com/weichen582/RetinexNet/blob/master/data/eval/low/1.png
- Checkpoint:
  https://huggingface.co/Fediory/HVI-CIDNet-LOLv2-real-bestPSNR
- Đây là phép thử chạy phần mềm; chưa phải kết quả đánh giá chất lượng hoặc
  bằng chứng về hạn chế của HVI. `Dataset/dataset.zip` bên cạnh repo là dữ liệu
  polyp và không được sử dụng cho phép thử này.

## Cài lại môi trường

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos.txt
```
