#!/usr/bin/env python3
import os
import glob
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np


class StereoImagePublisher(Node):
    def __init__(self,
                 left_dir: str = "left",
                 right_dir: str = "right",
                 fps: float = 10.0):
        super().__init__("stereo_image_publisher")

        self.left_dir = left_dir
        self.right_dir = right_dir
        self.period = 1.0 / fps

        # 加载、排序左右图像列表，保证数量一致且一一对应
        self.left_files = sorted(
            glob.glob(os.path.join(self.left_dir, "*.*"))
        )
        self.right_files = sorted(
            glob.glob(os.path.join(self.right_dir, "*.*"))
        )

        if not self.left_files:
            raise RuntimeError(f"左目目录中没有找到图像: {self.left_dir}")
        if not self.right_files:
            raise RuntimeError(f"右目目录中没有找到图像: {self.right_dir}")

        if len(self.left_files) != len(self.right_files):
            raise RuntimeError(
                f"左右图像数量不一致: left={len(self.left_files)}, right={len(self.right_files)}"
            )

        self.total_frames = len(self.left_files)
        self.current_index = 0

        # 创建发布者，话题名按你的要求
        self.left_pub = self.create_publisher(
            Image, "/image_left_raw/nv12_quarter_hengfortwoCamera2depthimage", 10
        )
        self.right_pub = self.create_publisher(
            Image, "/image_right_raw/nv12_quarter_hengfortwoCamera2depthimage", 10
        )

        # 定时器：每 0.1 秒发布一对
        self.timer = self.create_timer(self.period, self.timer_callback)

        self.get_logger().info(
            f"StereoImagePublisher 启动，左右各 {self.total_frames} 帧，"
            f"发布频率 {fps} Hz。"
        )

    def timer_callback(self):
        if self.current_index >= self.total_frames:
            # 播放完一轮后从第一帧重新开始循环
            self.get_logger().info("所有图像已发布完成，重新从第 1 帧开始循环播放。")
            self.current_index = 0

        left_path = self.left_files[self.current_index]
        right_path = self.right_files[self.current_index]

        # 读取图像（BGR）
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)

        if left_img is None:
            self.get_logger().error(f"读取左目图像失败: {left_path}")
            self.current_index += 1
            return

        if right_img is None:
            self.get_logger().error(f"读取右目图像失败: {right_path}")
            self.current_index += 1
            return

        # 确保左右尺寸一致
        if left_img.shape != right_img.shape:
            self.get_logger().error(
                f"左右图像尺寸不一致: left={left_img.shape}, right={right_img.shape}"
            )
            self.current_index += 1
            return

        stamp = self.get_clock().now().to_msg()

        left_msg = self.cv_image_to_msg(left_img, stamp, frame_id="left_camera")
        right_msg = self.cv_image_to_msg(right_img, stamp, frame_id="right_camera")

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

        self.get_logger().info(
            f"发布第 {self.current_index + 1}/{self.total_frames} 帧："
            f"{os.path.basename(left_path)} | {os.path.basename(right_path)}"
        )

        self.current_index += 1

    @staticmethod
    def cv_image_to_msg(cv_image: np.ndarray, stamp, frame_id: str) -> Image:
        """
        将 OpenCV BGR 图像转换为 ROS2 Image 消息。
        注意：这里 encoding 使用标准 'bgr8'，话题名里已经带有 nv12_quarter。
        如果你后续需要真正的 NV12 数据，可以在这里自己做格式转换。
        """
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        height, width, channels = cv_image.shape
        msg.height = height
        msg.width = width
        msg.encoding = "bgr8"  # 如果你一定要标记成 NV12，可改为 "nv12_quarter"
        msg.is_bigendian = False
        msg.step = width * channels
        msg.data = cv_image.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = StereoImagePublisher(
        left_dir="left",   # 如有需要可改为绝对路径
        right_dir="right", # 如有需要可改为绝对路径
        fps=10.0           # 0.1 秒一帧
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
