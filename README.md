# Cài đặt minBERT và cải tiến với phương pháp knowledge injection, contrastive learning
Thành viên:
- Tăng Vĩnh Hà, MSV: 22028129
- Vũ Nguyệt Hằng, MSV: 
- Ngô Tùng Lâm, MSV:

Repo cho bài tập lớn môn Xử lý ngôn ngữ tự nhiên, lớp 7 học kì 1 năm học 2024 - 2025. Chi tiết dự án ở trong báo cáo cuối kì. 

# Chi tiết dự án
- Cài đặt minBERT: 
    - File `tokenizer` và `bert_base` được dùng để tokenize input của câu đầu vào thành các input_ids theo bộ vocab có trước của google/bert_uncased_L-4_H-256_A-4 trên HuggingFace và khởi tạo tham số cho mô hình minBERT (phương thức `init_weights`).
    - File `bert`: cài đặt các thành phần của kiến trúc BERT (chi tiết giải thích được ghi chú trong phần comment).
    - File `classifier`: thêm các extra layer cho BERT từ mô hình BERT cho các downstream task cụ thể, trong project này là classify.
- Cải tiến với phương pháp knowledge injection:
- Cải tiến với phương pháp contrastive learning: nhóm tiếp tục pretrain BERT với SimCSE framework:
    - SimCSE unsupervised contrastive learning: sử dụng dữ liệu contrastive là các batch câu được forward qua model `bert` 2 lần (model sinh ra embedding) để được apply 2 `attention_mask` khác nhau cho positive pairs, các câu khác trong cùng 1 batch là negative pairs, hàm tối ưu là contrastive loss. Bộ dữ liệu sử dụng là: wiki1m (lấy 38% so với số lượng ban đầu). Được cài đặt trong file `classifier_unsupervised_CL`, chỉnh sửa từ file `classifier` ở chỗ load dữ liệu và hàm `train` (định nghĩa loss khác).
    - SimCSE supervised contrastive learning: sử dụng dữ liệu contrastive là bộ ba câu anchor - positive - negative trong bộ dữ liệu NLI, hàm tối ưu là contrastive loss. Được cài đặt trong file `classifier_supervised_CL`, chỉnh sửa từ file `classifier` ở chỗ load dữ liệu và hàm `train` (định nghĩa loss khác).
- Kết quả thử nghiệm: ở trong report cuối kì. 
- Cách chạy các thí nghiệm: vui lòng xem trong phần phụ lục notebook Kaggle của nhóm ở trong report cuối kì. 
# Cấu trúc repo
- Folder `data`: dữ liệu cho các thí nghiệm
- Folder `data_small`: dữ liệu cho việc debug nhanh trên laptop. 

# Acknowledgement
Nhóm tham khảo repo code gốc [`minBERT`](https://github.com/neubig/minbert-assignment).
