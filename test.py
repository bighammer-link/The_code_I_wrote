import cv2
import torch
import torch.nn.functional as F

def apply_pooling_and_save(input_path, output_avg_path, output_max_path,
                           kernel_size=3, stride=2):
    # 1. 读取图片（BGR）
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("无法读取输入图片，请检查路径。")

    # 转成 RGB 格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. HWC → CHW → tensor float32
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0)  # (1,3,H,W)

    # 3. 平均池化
    avg_pooled = F.avg_pool2d(tensor, kernel_size=kernel_size, stride=stride)

    # 4. 最大池化
    max_pooled = F.max_pool2d(tensor, kernel_size=kernel_size, stride=stride)

    # 5. 转为 numpy(HWC)，再保存
    avg_np = avg_pooled.squeeze(0).permute(1, 2, 0).byte().numpy()
    max_np = max_pooled.squeeze(0).permute(1, 2, 0).byte().numpy()

    # 保存时转回 BGR（OpenCV 用 BGR）
    cv2.imwrite(output_avg_path, cv2.cvtColor(avg_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_max_path, cv2.cvtColor(max_np, cv2.COLOR_RGB2BGR))

    print("平均池化输出保存至:", output_avg_path)
    print("最大池化输出保存至:", output_max_path)


# 示例调用
if __name__ == "__main__":
    apply_pooling_and_save(
        input_path="/home/jcob/personal/dogs.jpg",
        output_avg_path="/home/jcob/personal/avg_pool.png",
        output_max_path="/home/jcob/personal/max_pool.png",
        kernel_size=5,
        stride=3
    )
