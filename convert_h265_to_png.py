import os

import rclpy
from rclpy.node import Node
from foxglove_msgs.msg import CompressedVideo

from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

import av
from PIL import Image as PilImage


class StereoH265ToPngNode(Node):
    def __init__(self):
        super().__init__('stereo_h265_to_png_node')

        # 根据实际话题名修改
        self.right_topic = '/image_right_raw/h265'
        self.left_topic = '/image_left_raw/h265'

        # 右相机订阅 & 解码器
        self.sub_right = self.create_subscription(
            CompressedVideo,
            self.right_topic,
            self.right_callback,
            10
        )
        self.pub_right = self.create_publisher(
            RosImage,
            '/image_right_raw',
            10
        )
        self.codec_right = av.CodecContext.create('hevc', 'r')
        self.out_right = 'frames_right_png'
        os.makedirs(self.out_right, exist_ok=True)
        self.idx_right = 0

        # 左相机订阅 & 解码器
        self.sub_left = self.create_subscription(
            CompressedVideo,
            self.left_topic,
            self.left_callback,
            10
        )
        self.pub_left = self.create_publisher(
            RosImage,
            '/image_left_raw',
            10
        )
        self.codec_left = av.CodecContext.create('hevc', 'r')
        self.out_left = 'frames_left_png'
        os.makedirs(self.out_left, exist_ok=True)
        self.idx_left = 0

        self.bridge = CvBridge()

        self.get_logger().info(
            f'Subscribe right={self.right_topic}, left={self.left_topic}, format="h265"'
        )

    def decode_and_save(self, msg: CompressedVideo, codec_ctx: av.codec.context.CodecContext,
                        out_dir: str, idx_ref_name: str,
                        publisher: rclpy.node.Publisher, frame_id: str):
        if msg.format.lower() != 'h265':
            self.get_logger().warn(f'Ignore frame with format="{msg.format}"')
            return

        packet = av.Packet(bytes(msg.data))

        try:
            frames = codec_ctx.decode(packet)
        except av.AVError as e:
            self.get_logger().error(f'FFmpeg decode error: {e}')
            return

        # 通过属性名存取左右各自的计数器
        idx = getattr(self, idx_ref_name)

        for frame in frames:
            # 转成 OpenCV BGR 图像，先发布到 ROS2 话题，再保存 PNG
            cv_image = frame.to_ndarray(format='bgr24')
            height, width, _ = cv_image.shape

            # 构造 ROS2 Image 消息并发布
            img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            img_msg.header.stamp = msg.timestamp
            img_msg.header.frame_id = frame_id

            # 在终端输出图像基本信息：格式和尺寸
            self.get_logger().info(
                f'Publish Image: encoding="bgr8", size={width}x{height}, '
                f'topic="{publisher.topic_name}"'
            )

            publisher.publish(img_msg)

            # 同时也保存 PNG 到磁盘
            img = PilImage.fromarray(cv_image[:, :, ::-1])  # BGR -> RGB
            t_sec = msg.timestamp.sec
            t_nsec = msg.timestamp.nanosec
            filename = os.path.join(
                out_dir,
                f'{t_sec}_{t_nsec}_{idx:06d}.png'
            )
            img.save(filename)
            self.get_logger().info(f'Saved {filename}')
            idx += 1

        setattr(self, idx_ref_name, idx)

    def right_callback(self, msg: CompressedVideo):
        self.decode_and_save(
            msg,
            self.codec_right,
            self.out_right,
            'idx_right',
            self.pub_right,
            'camera_right'
        )

    def left_callback(self, msg: CompressedVideo):
        self.decode_and_save(
            msg,
            self.codec_left,
            self.out_left,
            'idx_left',
            self.pub_left,
            'camera_left'
        )


def main(args=None):
    rclpy.init(args=args)
    node = StereoH265ToPngNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
