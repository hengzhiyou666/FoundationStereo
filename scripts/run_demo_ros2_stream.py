import os
import sys
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo


def _rosimg_to_bgr(img_msg: RosImage) -> np.ndarray:
  """
  将 ROS2 Image 消息转换为 BGR numpy 图像。
  假设编码是 bgr8 / rgb8 / mono8。
  """
  h, w = img_msg.height, img_msg.width
  data = np.frombuffer(img_msg.data, dtype=np.uint8)

  if img_msg.encoding.lower() in ["mono8", "8uc1"]:
    img = data.reshape(h, w, 1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

  img = data.reshape(h, w, -1)
  if img_msg.encoding.lower() in ["rgb8", "rgb"]:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  return img


class StereoInferenceNode(Node):
  """
  持续从左右图像话题读取图像，并使用 FoundationStereo 推理，实时更新 vis 窗口。
  """

  def __init__(self, args):
    super().__init__("foundation_stereo_ros2_stream")
    self.args = args

    # 初始化模型
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
      cfg["vit_size"] = "vitl"
    for k in vars(args):
      cfg[k] = getattr(args, k)
    cfg = OmegaConf.create(cfg)

    logging.info(f"args:\n{cfg}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    self.model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    self.model.load_state_dict(ckpt["model"])
    self.model.cuda()
    self.model.eval()

    # 相机内参 / 基线（用于深度）
    with open(args.intrinsic_file, "r") as f:
      lines = f.readlines()
      self.K = (
        np.array(list(map(float, lines[0].rstrip().split())))
        .astype(np.float32)
        .reshape(3, 3)
      )
      self.baseline = float(lines[1])

    self.left_topic = args.left_topic
    self.right_topic = args.right_topic
    self.scale = args.scale

    self.left_msg = None
    self.right_msg = None
    self.last_stamp = None
    self.busy = False

    self.left_sub = self.create_subscription(
      RosImage, self.left_topic, self._left_cb, 10
    )
    self.right_sub = self.create_subscription(
      RosImage, self.right_topic, self._right_cb, 10
    )

    self.get_logger().info(
      f"StereoInferenceNode started. Subscribing to:\n"
      f"  left : {self.left_topic}\n"
      f"  right: {self.right_topic}\n"
      f"结果窗口将持续更新，按 Ctrl+C 结束。"
    )

  def _left_cb(self, msg: RosImage):
    self.left_msg = msg
    self._try_process()

  def _right_cb(self, msg: RosImage):
    self.right_msg = msg
    self._try_process()

  def _try_process(self):
    if self.busy:
      return
    if self.left_msg is None or self.right_msg is None:
      return

    # 要求左右帧时间戳一致（你自己的发布节点已经保证）
    ls = self.left_msg.header.stamp
    rs = self.right_msg.header.stamp
    if ls.sec != rs.sec or ls.nanosec != rs.nanosec:
      return

    if self.last_stamp is not None:
      if ls.sec == self.last_stamp.sec and ls.nanosec == self.last_stamp.nanosec:
        return

    self.last_stamp = ls

    try:
      self.busy = True
      img0 = _rosimg_to_bgr(self.left_msg)
      img1 = _rosimg_to_bgr(self.right_msg)
      self._run_inference_and_show(img0, img1)
    except Exception as e:
      self.get_logger().error(f"推理失败: {e}")
    finally:
      self.busy = False

  def _run_inference_and_show(self, img0: np.ndarray, img1: np.ndarray):
    # 与 scripts/run_demo.py 中逻辑保持一致
    if img0.ndim == 3 and img0.shape[2] == 4:
      img0 = img0[..., :3]
    if img1.ndim == 3 and img1.shape[2] == 4:
      img1 = img1[..., :3]

    scale = self.scale
    assert scale <= 1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    if torch.cuda.is_available():
      torch.cuda.synchronize()
    start_time = time.time()
    with torch.cuda.amp.autocast(True):
      if not self.args.hiera:
        disp = self.model.forward(
          img0_t, img1_t, iters=self.args.valid_iters, test_mode=True
        )
      else:
        disp = self.model.run_hierachical(
          img0_t,
          img1_t,
          iters=self.args.valid_iters,
          test_mode=True,
          small_ratio=0.5,
        )
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    infer_time = time.time() - start_time
    logging.info(f"[stream] Inference time: {infer_time:.3f} s")

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)

    # 覆盖写一份 vis.png，方便你随时查看磁盘文件
    os.makedirs(self.args.out_dir, exist_ok=True)
    vis_path = os.path.join(self.args.out_dir, "vis.png")
    import imageio
    imageio.imwrite(vis_path, vis)

    # 显示窗口，实时更新
    try:
      cv2.imshow("FoundationStereo vis (stream)", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
      cv2.waitKey(1)
    except Exception as e:
      logging.warning(f"Failed to show vis window: {e}")

    # 可选：计算深度和点云（同 run_demo 的 get_pc / save_pc 逻辑）
    if self.args.get_pc:
      K = self.K.copy()
      K[:2] *= scale
      depth = K[0, 0] * self.baseline / disp
      np.save(os.path.join(self.args.out_dir, "depth_meter.npy"), depth)

      if self.args.save_pc:
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (
          np.asarray(pcd.points)[:, 2] <= self.args.z_far
        )
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        import open3d as o3d

        o3d.io.write_point_cloud(
          os.path.join(self.args.out_dir, "cloud.ply"), pcd
        )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--intrinsic_file",
    default=f"{code_dir}/../assets/K.txt",
    type=str,
    help="camera intrinsic matrix and baseline file",
  )
  parser.add_argument(
    "--ckpt_dir",
    default=f"{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth",
    type=str,
    help="pretrained model path",
  )
  parser.add_argument(
    "--out_dir",
    default=f"{code_dir}/../output_stream/",
    type=str,
    help="directory to save outputs (vis.png, depth, etc.)",
  )
  parser.add_argument(
    "--scale",
    default=1.0,
    type=float,
    help="downsize the image by scale, must be <=1",
  )
  parser.add_argument(
    "--hiera",
    default=0,
    type=int,
    help="hierarchical inference (for high-res images)",
  )
  parser.add_argument(
    "--z_far",
    default=10.0,
    type=float,
    help="max depth to clip in point cloud",
  )
  parser.add_argument(
    "--valid_iters",
    type=int,
    default=32,
    help="number of flow-field updates during forward pass",
  )
  parser.add_argument(
    "--get_pc",
    type=int,
    default=0,
    help="为 1 时读内参、算深度并保存 depth_meter.npy（流式模式下按需使用）",
  )
  parser.add_argument(
    "--save_pc",
    type=int,
    default=0,
    help="为 1 时才生成并保存 cloud.ply（需 get_pc=1）",
  )
  parser.add_argument(
    "--left_topic",
    type=str,
    default="/image_left_raw/nv12_quarter_hengfortwoCamera2depthimage",
    help="左目图像话题名",
  )
  parser.add_argument(
    "--right_topic",
    type=str,
    default="/image_right_raw/nv12_quarter_hengfortwoCamera2depthimage",
    help="右目图像话题名",
  )

  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  rclpy.init(args=None)
  node = StereoInferenceNode(args)
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()
    try:
      cv2.destroyAllWindows()
    except Exception:
      pass


if __name__ == "__main__":
  main()

