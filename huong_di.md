# BÁO CÁO ĐỀ XUẤT HƯỚNG NGHIÊN CỨU

## Đề tài: Nghiên cứu, phát triển hệ thống giám sát video thời gian thực bảo vệ quyền riêng tư trên thiết bị biên nhúng cho UAV

## 1. Tổng quan đề tài

### 1.1. Bối cảnh nghiên cứu

UAV (Unmanned Aerial Vehicle) được sử dụng trong các nhiệm vụ giám sát, quan sát và thu thập dữ liệu hình ảnh từ môi trường thực tế. Video thu được có thể chứa thông tin nhận dạng cá nhân, chẳng hạn như khuôn mặt và biển số phương tiện. Việc lưu trữ hoặc truyền tải video chưa được bảo vệ có thể dẫn đến nguy cơ tiết lộ thông tin riêng tư.

Một phương pháp bảo vệ quyền riêng tư là ẩn danh các vùng hình ảnh chứa thông tin nhạy cảm trước khi video được xuất hoặc truyền đi. Tuy nhiên, trong hệ thống giám sát UAV, quá trình xử lý cần đáp ứng yêu cầu thời gian thực và giới hạn tài nguyên tính toán của thiết bị biên nhúng.

Nếu áp dụng một mức ẩn danh cố định cho mọi đối tượng và mọi tình huống, hệ thống sẽ không phân biệt được sự khác nhau về mức độ rủi ro riêng tư giữa các ngữ cảnh quan sát. Vì vậy, nghiên cứu tập trung vào việc đánh giá rủi ro quyền riêng tư theo ngữ cảnh và sử dụng kết quả đánh giá để lựa chọn mức độ ẩn danh phù hợp.

### 1.2. Vấn đề nghiên cứu

Đề tài tập trung giải quyết ba vấn đề chính:

1. Phát hiện và theo dõi các đối tượng chứa thông tin có khả năng nhận dạng cá nhân trong video.
2. Đánh giá rủi ro quyền riêng tư dựa trên thông tin đối tượng và ngữ cảnh quan sát.
3. Tích hợp cơ chế lựa chọn mức độ ẩn danh vào pipeline xử lý video thời gian thực trên thiết bị biên nhúng.

Hệ thống phải kiểm soát video đầu ra để tránh truyền các khung hình chưa đáp ứng điều kiện bảo vệ quyền riêng tư đã được thiết lập.

## 2. Mục tiêu nghiên cứu

### 2.1. Mục tiêu tổng quát

Nghiên cứu và phát triển hệ thống xử lý video thời gian thực trên thiết bị biên nhúng cho UAV, trong đó mức độ ẩn danh được lựa chọn dựa trên kết quả đánh giá rủi ro quyền riêng tư theo ngữ cảnh.

### 2.2. Mục tiêu cụ thể

- Xây dựng pipeline xử lý video gồm phát hiện đối tượng, theo dõi đối tượng, đánh giá rủi ro và ẩn danh.
- Xây dựng cơ chế biểu diễn và đánh giá rủi ro quyền riêng tư theo ngữ cảnh.
- Xây dựng bộ máy chính sách ánh xạ mức rủi ro thành mức độ bảo vệ tương ứng.
- Thiết kế cơ chế kiểm tra video trước khi xuất hoặc truyền.
- Đánh giá độ chính xác của các mô-đun, hiệu quả bảo vệ quyền riêng tư và hiệu năng xử lý trên thiết bị biên nhúng.

## 3. Phạm vi nghiên cứu

Đề tài tập trung vào bảo vệ quyền riêng tư trong video thu nhận từ UAV, với khuôn mặt và biển số phương tiện là các loại thông tin nhận dạng được xem xét trong phạm vi nghiên cứu.

Hệ thống xử lý video theo từng khung hình, kết hợp thông tin theo thời gian từ mô-đun theo dõi đối tượng và sử dụng ngữ cảnh để xác định mức độ ẩn danh.

Phạm vi đánh giá bao gồm:

- Khả năng phát hiện đối tượng cần bảo vệ.
- Khả năng duy trì theo dõi đối tượng giữa các khung hình.
- Khả năng đánh giá rủi ro quyền riêng tư.
- Khả năng thực thi chính sách ẩn danh.
- Khả năng ngăn khung hình chưa đáp ứng điều kiện bảo vệ đi vào luồng đầu ra.
- Hiệu năng xử lý video trên thiết bị biên nhúng.

Đề tài không mặc định rằng việc phát hiện khuôn mặt và biển số bao phủ toàn bộ nguy cơ riêng tư. Những loại thông tin ngoài phạm vi phát hiện đã triển khai không được xem là đã được bảo vệ.

## 4. Kiến trúc tổng thể của hệ thống

Pipeline xử lý được tổ chức theo trình tự:

**Thu nhận video → Tiền xử lý → Phát hiện đối tượng → Theo dõi đối tượng → Trích xuất đặc trưng ngữ cảnh → Đánh giá rủi ro quyền riêng tư → Bộ máy chính sách → Ẩn danh → Kiểm tra bảo vệ → Mã hóa và truyền video.**

Các mô-đun được mô tả như sau.

### 4.1. Mô-đun thu nhận video

Tiếp nhận luồng video từ camera UAV, quản lý khung hình, dấu thời gian và bộ đệm phục vụ xử lý.

Dữ liệu đầu vào của mô-đun là luồng video; đầu ra là các khung hình được chuyển đến pipeline xử lý.

### 4.2. Mô-đun tiền xử lý

Chuẩn hóa kích thước và định dạng khung hình theo yêu cầu của mô hình phát hiện đối tượng. Mô-đun thực hiện các thao tác cần thiết để đưa dữ liệu vào mô hình suy luận.

Các thông số tiền xử lý được xác định theo cấu hình mô hình và độ phân giải video sử dụng trong thực nghiệm.

### 4.3. Mô-đun phát hiện đối tượng

Phát hiện các đối tượng thuộc phạm vi bảo vệ, gồm khuôn mặt và biển số phương tiện.

Đầu ra của mô-đun bao gồm loại đối tượng, tọa độ vùng phát hiện và độ tin cậy của kết quả. Những thông tin này được chuyển sang mô-đun theo dõi và đánh giá rủi ro.

Độ tin cậy phát hiện được sử dụng như một thông tin phục vụ kiểm soát kết quả, không được xem là bằng chứng độc lập cho thấy đối tượng đã được bảo vệ đầy đủ.

### 4.4. Mô-đun theo dõi đối tượng

Liên kết các kết quả phát hiện giữa những khung hình liên tiếp nhằm duy trì thông tin về đối tượng theo thời gian.

Thông tin theo dõi bao gồm mã định danh theo dõi, vị trí đối tượng, thời gian xuất hiện và trạng thái theo dõi. Các thông tin này phục vụ việc xác định thời gian quan sát và hỗ trợ đánh giá rủi ro tích lũy.

Khi đối tượng bị mất dấu hoặc trạng thái theo dõi không đáng tin cậy, hệ thống phải ghi nhận trạng thái bất định để xử lý theo chính sách bảo vệ đã xác định.

### 4.5. Mô-đun trích xuất đặc trưng ngữ cảnh

Tổng hợp thông tin từ mô-đun phát hiện, mô-đun theo dõi và dữ liệu ngữ cảnh có sẵn.

Các đặc trưng thuộc phạm vi thiết kế bao gồm:

- Loại đối tượng cần bảo vệ.
- Độ tin cậy phát hiện.
- Trạng thái theo dõi.
- Thời gian đối tượng xuất hiện trong khung hình hoặc chuỗi khung hình.
- Tình trạng che khuất hoặc mất dấu, khi có thể xác định.
- Thông tin khu vực quan sát, nếu hệ thống có dữ liệu vị trí và quy tắc phân vùng tương ứng.

Mỗi đặc trưng chỉ được sử dụng khi có dữ liệu đầu vào và quy tắc diễn giải xác định. Hệ thống không tự suy diễn loại khu vực hoặc mức độ nhạy cảm nếu không có căn cứ dữ liệu.

### 4.6. Mô-đun đánh giá rủi ro quyền riêng tư

Mô-đun tiếp nhận đặc trưng đối tượng và ngữ cảnh để xác định mức rủi ro quyền riêng tư.

Kết quả đánh giá được sử dụng làm đầu vào cho bộ máy chính sách. Cách biểu diễn rủi ro, các biến đầu vào, trọng số và ngưỡng phân loại phải được xác định cụ thể trong phương pháp thực nghiệm.

Mô-đun phải phân biệt giữa rủi ro được đánh giá và độ bất định của hệ thống. Trường hợp dữ liệu phát hiện hoặc theo dõi không đủ tin cậy không được mặc nhiên xem là trường hợp có rủi ro thấp.

### 4.7. Mô-đun thực thi chính sách bảo vệ

Bộ máy chính sách nhận kết quả đánh giá rủi ro và xác định mức độ ẩn danh cần áp dụng.

Chính sách phải quy định rõ quan hệ giữa mức rủi ro, trạng thái bất định và mức bảo vệ. Khi thông tin đầu vào không đủ để đưa ra quyết định đáng tin cậy, hệ thống phải áp dụng quy tắc dự phòng đã được xác định trước, thay vì tự động chọn mức bảo vệ thấp.

Các mức bảo vệ, điều kiện chuyển mức và ngưỡng quyết định được xác định trong thiết kế thực nghiệm; không mặc định các ngưỡng khi chưa có căn cứ.

### 4.8. Mô-đun ẩn danh

Thực hiện biến đổi vùng hình ảnh cần bảo vệ theo chính sách đã lựa chọn.

Phương pháp ẩn danh được đánh giá theo khả năng giảm khả năng nhận dạng của thông tin nhạy cảm và mức ảnh hưởng đến chất lượng hình ảnh còn lại. Các chỉ số chất lượng ảnh không được sử dụng thay thế cho phép đo trực tiếp về khả năng nhận dạng.

### 4.9. Mô-đun kiểm tra bảo vệ trước đầu ra

Mô-đun kiểm tra điều kiện bảo vệ của khung hình sau xử lý trước khi đưa vào luồng đầu ra.

Các điều kiện kiểm tra bao gồm việc thực thi chính sách ẩn danh, tính hợp lệ của vùng xử lý và trạng thái xử lý của khung hình theo thiết kế hệ thống.

Nếu khung hình không đáp ứng điều kiện đầu ra, hệ thống không được chuyển khung hình đó đi như một khung hình đã được bảo vệ. Hành vi dự phòng phải được quy định trong thiết kế.

Mô-đun này không thể tự bảo đảm phát hiện mọi đối tượng mà mô-đun phát hiện ban đầu đã bỏ sót. Vì vậy, khả năng bảo vệ tổng thể vẫn phụ thuộc vào độ bao phủ phát hiện, tính đúng đắn của theo dõi và chính sách xử lý.

### 4.10. Mô-đun mã hóa và truyền video

Chỉ tiếp nhận các khung hình đã vượt qua điều kiện kiểm tra bảo vệ để thực hiện mã hóa và truyền video.

Pipeline phải bảo đảm thứ tự xử lý giữa ẩn danh, kiểm tra bảo vệ và truyền tải. Việc đánh giá hiệu năng sử dụng thời gian xử lý đầu-cuối, không chỉ thời gian suy luận của mô hình phát hiện.

### 4.11. Quản lý tài nguyên tính toán

Mô-đun quản lý tài nguyên theo dõi các thông số vận hành như tốc độ khung hình, độ trễ, mức sử dụng bộ xử lý, bộ nhớ và công suất tiêu thụ khi thiết bị cung cấp dữ liệu đo tương ứng.

Mục tiêu là đánh giá khả năng vận hành thời gian thực trong giới hạn tài nguyên của thiết bị. Các điều chỉnh hiệu năng không được làm vô hiệu hóa điều kiện bảo vệ quyền riêng tư đã xác định.

## 5. Phương pháp đánh giá rủi ro quyền riêng tư

### 5.1. Đầu vào đánh giá

Đầu vào gồm loại đối tượng, thông tin phát hiện, trạng thái theo dõi, thời gian quan sát và các đặc trưng ngữ cảnh có dữ liệu hỗ trợ.

Các đặc trưng phải được chuẩn hóa hoặc chuyển đổi theo quy tắc được mô tả rõ trong phương pháp thực nghiệm để bảo đảm kết quả có thể tái lập.

### 5.2. Biểu diễn rủi ro

Rủi ro quyền riêng tư được sử dụng để phân loại tình huống theo các mức độ bảo vệ. Phương pháp tính điểm hoặc mô hình phân loại phải được xác định cùng với:

- Danh sách đặc trưng đầu vào.
- Quy tắc gán nhãn rủi ro.
- Quy tắc chuẩn hóa đặc trưng.
- Cách xác định trọng số hoặc tham số mô hình.
- Ngưỡng phân loại.
- Quy tắc xử lý dữ liệu thiếu và trạng thái bất định.

Các thông số này chưa được xem là kết quả đã xác lập nếu chưa có quy trình thiết kế, hiệu chỉnh và đánh giá tương ứng.

### 5.3. Quan hệ giữa rủi ro và mức ẩn danh

Bộ máy chính sách sử dụng kết quả đánh giá để lựa chọn mức độ ẩn danh. Quan hệ ánh xạ phải được thể hiện bằng bảng chính sách cụ thể, trong đó quy định hành vi của hệ thống cho từng mức rủi ro và trạng thái bất định.

Tính đúng đắn của chính sách được đánh giá thông qua dữ liệu gán nhãn, kết quả thực thi và các chỉ số đo lường. Không kết luận rằng cơ chế thích ứng hiệu quả hơn ẩn danh cố định trước khi hoàn thành so sánh thực nghiệm.

## 6. Phương pháp và công nghệ triển khai

### 6.1. Môi trường phần mềm

Hệ thống được triển khai trên môi trường Linux của thiết bị đích. Các thành phần xử lý ảnh, suy luận mô hình và truyền video phải được tích hợp thành pipeline có thể đo thời gian xử lý đầu-cuối.

Các công nghệ chỉ được ghi là công nghệ sử dụng chính thức khi cấu hình triển khai đã được xác định và kiểm thử. Phiên bản phần mềm, thư viện và tham số thực thi phải được lưu lại để phục vụ tái lập thực nghiệm.

### 6.2. Mô hình phát hiện và theo dõi

Mô hình phát hiện phải đáp ứng nhiệm vụ nhận diện các đối tượng thuộc phạm vi bảo vệ. Mô-đun theo dõi phải duy trì liên kết đối tượng giữa các khung hình.

Tên mô hình, phiên bản, trọng số, cấu hình đầu vào và phương pháp theo dõi được ghi trong hồ sơ triển khai sau khi lựa chọn. Không xem một mô hình cụ thể là phương án cuối cùng khi chưa có kết quả kiểm thử trên dữ liệu và phần cứng mục tiêu.

### 6.3. Thiết bị biên nhúng

Thiết bị triển khai phải được xác định rõ về bộ xử lý, bộ nhớ, bộ tăng tốc suy luận nếu có và giới hạn công suất.

Kết quả đánh giá hiệu năng phải gắn với thiết bị cụ thể, độ phân giải video, cấu hình mô hình và điều kiện tải. Không ngoại suy kết quả đo trên một thiết bị thành hiệu năng của các thiết bị khác khi chưa có dữ liệu xác nhận.

## 7. Thiết kế thực nghiệm

### 7.1. Mục đích thực nghiệm

Thực nghiệm nhằm xác định:

1. Độ chính xác của phát hiện và theo dõi đối tượng cần bảo vệ.
2. Khả năng phân loại rủi ro quyền riêng tư theo ngữ cảnh.
3. Hiệu quả giảm khả năng nhận dạng sau ẩn danh.
4. Hiệu năng xử lý video trên thiết bị biên nhúng.
5. Tác động của cơ chế đánh giá rủi ro và chính sách thích ứng so với phương pháp ẩn danh cố định.

### 7.2. Các cấu hình so sánh

Thực nghiệm sử dụng các cấu hình sau khi được triển khai đầy đủ:

- **Cấu hình A:** Phát hiện đối tượng và ẩn danh cố định.
- **Cấu hình B:** Phát hiện, theo dõi đối tượng và ẩn danh cố định.
- **Cấu hình C:** Phát hiện, theo dõi, đánh giá rủi ro theo ngữ cảnh, thực thi chính sách và ẩn danh thích ứng.

Các cấu hình phải được đánh giá trên cùng tập dữ liệu và trong điều kiện so sánh tương đương. Những khác biệt về độ phân giải, phần cứng, cấu hình suy luận hoặc điều kiện tải phải được ghi rõ.

### 7.3. Dữ liệu thực nghiệm

Tập dữ liệu phải có thông tin nhãn phù hợp với nhiệm vụ được đánh giá. Đối với phát hiện và theo dõi, cần có nhãn đối tượng và thông tin liên kết theo thời gian tương ứng. Đối với đánh giá rủi ro, cần có quy tắc gán nhãn và nhãn rủi ro nhất quán.

Nguồn dữ liệu, giấy phép sử dụng, điều kiện thu thập và cách chia tập huấn luyện, hiệu chỉnh và kiểm thử phải được ghi trong báo cáo thực nghiệm.

Tên bộ dữ liệu chính thức chỉ được đưa vào danh mục triển khai sau khi đã kiểm tra tính phù hợp với bối cảnh UAV, phạm vi đối tượng và yêu cầu đánh giá.

### 7.4. Quy trình đánh giá

1. Kiểm tra tính hợp lệ của dữ liệu và nhãn.
2. Đánh giá riêng mô-đun phát hiện đối tượng.
3. Đánh giá mô-đun theo dõi đối tượng.
4. Đánh giá mô-đun phân loại rủi ro trên dữ liệu có nhãn.
5. Đánh giá khả năng giảm nhận dạng của video sau ẩn danh.
6. Đo hiệu năng pipeline hoàn chỉnh trên thiết bị mục tiêu.
7. So sánh các cấu hình theo cùng giao thức thực nghiệm.
8. Phân tích lỗi và xác định các tình huống hệ thống chưa đáp ứng yêu cầu.

## 8. Các chỉ số đánh giá

### 8.1. Phát hiện đối tượng

- Precision.
- Recall.
- mAP@0.5.
- mAP@0.5:0.95.

Recall được sử dụng để đánh giá mức độ bao phủ đối tượng cần bảo vệ. Đối tượng bị bỏ sót có thể không được ẩn danh, vì vậy kết quả phát hiện phải được phân tích cùng kết quả bảo vệ cuối cùng.

### 8.2. Theo dõi đối tượng

- HOTA.
- IDF1.
- Số lần thay đổi định danh đối tượng (ID switches).

Các chỉ số được báo cáo khi dữ liệu có nhãn theo dõi phù hợp với giao thức đánh giá.

### 8.3. Đánh giá rủi ro

- Accuracy.
- Precision.
- Recall.
- Macro-F1.
- Ma trận nhầm lẫn.

Recall của lớp rủi ro cao phải được báo cáo riêng để xác định tỷ lệ tình huống rủi ro cao bị phân loại nhầm thành mức thấp hơn.

### 8.4. Hiệu quả bảo vệ quyền riêng tư

Sử dụng phép đánh giá nhận dạng phù hợp với loại thông tin được bảo vệ. Đối với khuôn mặt, có thể sử dụng tỷ lệ xác minh đúng tại một mức tỷ lệ chấp nhận sai cố định (TAR tại FAR cố định), với cùng giao thức so sánh trước và sau ẩn danh.

Kết quả phải thể hiện khả năng nhận dạng thay đổi như thế nào sau xử lý. SSIM và PSNR chỉ phản ánh một số đặc tính chất lượng ảnh, không đủ để kết luận rằng thông tin cá nhân đã được bảo vệ.

### 8.5. Hiệu năng trên thiết bị biên

- Tốc độ xử lý video (FPS).
- Độ trễ đầu-cuối.
- Độ trễ phân vị 95 (P95 latency).
- Công suất tiêu thụ (W), nếu đo được.
- Năng lượng trên mỗi khung hình (J/frame), nếu có dữ liệu công suất và thời gian đo phù hợp.
- Mức sử dụng bộ nhớ cực đại.

Các phép đo phải bao gồm pipeline từ đầu vào đến đầu ra được bảo vệ. Không sử dụng FPS của riêng mô hình suy luận để đại diện cho tốc độ toàn hệ thống.

## 9. Bảng ghi nhận kết quả thực nghiệm

Kết quả được điền sau khi hoàn thành đo kiểm. Không sử dụng số liệu ước lượng thay cho số liệu thực nghiệm.

| Nhóm đánh giá | Chỉ số | Kết quả |
|---|---|---|
| Phát hiện | Precision, Recall, mAP@0.5, mAP@0.5:0.95 | Chưa đo |
| Theo dõi | HOTA, IDF1, ID switches | Chưa đo |
| Đánh giá rủi ro | Accuracy, Macro-F1, Recall lớp rủi ro cao | Chưa đo |
| Bảo vệ quyền riêng tư | TAR tại FAR cố định hoặc chỉ số phù hợp với nhiệm vụ | Chưa đo |
| Hiệu năng | FPS, độ trễ đầu-cuối, P95 latency | Chưa đo |
| Tài nguyên | Bộ nhớ cực đại, công suất, năng lượng/khung hình nếu đo được | Chưa đo |
| So sánh phương pháp | Chênh lệch giữa cấu hình A, B và C | Chưa đánh giá |

## 10. Các yêu cầu kỹ thuật cần kiểm chứng

### 10.1. Tính đúng đắn của phát hiện và theo dõi

Hệ thống cần được đánh giá trong các tình huống có thay đổi góc nhìn, kích thước đối tượng, chuyển động, che khuất và điều kiện hình ảnh khác nhau. Phạm vi thử nghiệm cụ thể được ghi nhận dựa trên dữ liệu thực tế sử dụng.

### 10.2. Tính nhất quán của chính sách

Cùng một tình huống đầu vào và cùng một cấu hình chính sách phải tạo ra quyết định nhất quán theo quy tắc đã xác định. Những trường hợp dữ liệu thiếu hoặc trạng thái bất định phải có hành vi xử lý được mô tả rõ.

### 10.3. Tính toàn vẹn của luồng đầu ra

Video truyền đi phải tuân thủ thứ tự xử lý: phát hiện, đánh giá, ẩn danh, kiểm tra bảo vệ và xuất video. Cần kiểm thử các trường hợp lỗi nhằm xác định hệ thống có vô tình chuyển tiếp khung hình chưa được bảo vệ hay không.

### 10.4. Khả năng vận hành thời gian thực

Hiệu năng được xác định thông qua đo kiểm trên thiết bị thực tế. Báo cáo phải ghi rõ cấu hình phần cứng, phần mềm, độ phân giải, điều kiện tải và phương pháp đo.

Không kết luận hệ thống đáp ứng thời gian thực nếu chưa xác định yêu cầu FPS, giới hạn độ trễ và chưa có kết quả đo tương ứng.

## 11. Giới hạn của phương pháp

Độ bảo vệ của hệ thống phụ thuộc vào khả năng phát hiện đối tượng, tính chính xác của thông tin theo dõi, độ đầy đủ của dữ liệu ngữ cảnh và tính đúng đắn của chính sách ẩn danh.

Mô-đun kiểm tra đầu ra chỉ có thể kiểm tra các điều kiện đã được thiết kế và triển khai; nó không thay thế được việc đánh giá độ bao phủ của mô-đun phát hiện.

Các chỉ số hiệu năng và độ chính xác không tự động chứng minh hệ thống bảo vệ được mọi dạng thông tin riêng tư. Kết luận phải giới hạn trong đối tượng, dữ liệu, điều kiện thử nghiệm và giao thức đánh giá đã sử dụng.

## 12. Kết luận

Đề tài tập trung phát triển hệ thống giám sát video thời gian thực trên thiết bị biên nhúng cho UAV, trong đó mức độ ẩn danh được lựa chọn dựa trên đánh giá rủi ro quyền riêng tư theo ngữ cảnh.

Kiến trúc hệ thống kết hợp phát hiện đối tượng, theo dõi theo thời gian, trích xuất đặc trưng ngữ cảnh, đánh giá rủi ro, thực thi chính sách, ẩn danh và kiểm tra trước khi truyền video. Việc đánh giá được thực hiện trên ba phương diện chính: độ chính xác của các mô-đun, hiệu quả bảo vệ quyền riêng tư và hiệu năng xử lý trên thiết bị đích.

Tại thời điểm lập báo cáo, các kết quả về độ chính xác, mức giảm khả năng nhận dạng, tốc độ xử lý và mức tiêu thụ tài nguyên chưa được xác lập bằng thực nghiệm. Vì vậy, hiệu quả của phương pháp và mức độ đáp ứng yêu cầu thời gian thực chỉ được kết luận sau khi hoàn thành thiết kế, triển khai và đánh giá theo các tiêu chí đã nêu.