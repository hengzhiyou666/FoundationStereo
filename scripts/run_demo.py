# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import argparse
import imageio
import torch
import logging
import cv2
import numpy as np
import open3d as o3d
import time
from typing import Optional, Tuple

try:
  import rclpy
  from rclpy.node import Node
  from sensor_msgs.msg import Image as RosImage
  _ROS2_AVAILABLE = True
except ImportError:
  _ROS2_AVAILABLE = False

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo


def _rosimg_to_bgr(img_msg: RosImage) -> np.ndarray:
  """
  将 ROS2 Image 消息转换为 BGR numpy 图像。
  假设编码是 bgr8 或 rgb8 或 mono8（灰度）。
  """
  h, w = img_msg.height, img_msg.width
  data = np.frombuffer(img_msg.data, dtype=np.uint8)

  # mono8
  if img_msg.encoding.lower() in ["mono8", "8uc1"]:
    img = data.reshape(h, w, 1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

  # bgr8 / rgb8
  img = data.reshape(h, w, -1)
  if img_msg.encoding.lower() in ["rgb8", "rgb"]:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  # 如果已经是 bgr8，就直接返回
  return img


class _StereoImageSubscriber(Node):
  """简单订阅器：各收一帧左右图后退出。"""

  def __init__(self, left_topic: str, right_topic: str):
    super().__init__('foundation_stereo_input_node')
    self._left: Optional[np.ndarray] = None
    self._right: Optional[np.ndarray] = None
    self._left_sub = self.create_subscription(RosImage, left_topic, self._left_cb, 10)
    self._right_sub = self.create_subscription(RosImage, right_topic, self._right_cb, 10)

  def _left_cb(self, msg: RosImage):
    try:
      self._left = _rosimg_to_bgr(msg)
    except Exception as e:
      self.get_logger().error(f"左目图像解析失败: {e}")

  def _right_cb(self, msg: RosImage):
    try:
      self._right = _rosimg_to_bgr(msg)
    except Exception as e:
      self.get_logger().error(f"右目图像解析失败: {e}")

  def get_pair(self, timeout: float) -> Tuple[np.ndarray, np.ndarray]:
    start = time.time()
    while (self._left is None or self._right is None) and (time.time() - start) < timeout:
      rclpy.spin_once(self, timeout_sec=0.1)
    if self._left is None or self._right is None:
      raise TimeoutError("在指定超时时间内没有同时收到左右图像。")
    return self._left, self._right


def get_input_images(args) -> Tuple[np.ndarray, np.ndarray]:
  """根据参数选择从文件或 ROS2 读取一对图像。"""
  if getattr(args, "use_ros2", False):
    if not _ROS2_AVAILABLE:
      raise RuntimeError("检测到 use_ros2=1，但当前 Python 环境没有安装 rclpy / sensor_msgs。请在 ROS2 Humble 环境中运行。")
    rclpy.init(args=None)
    try:
      node = _StereoImageSubscriber(args.left_topic, args.right_topic)
      img0, img1 = node.get_pair(args.ros2_timeout)
    finally:
      rclpy.shutdown()
    return img0, img1

  # 默认：从文件读取
  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  return img0, img1


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='为 1 时读内参、算深度并保存 depth_meter.npy')
  parser.add_argument('--save_pc', type=int, default=0, help='为 1 时才生成并保存 cloud.ply、cloud_denoise.ply 及点云窗口（需 get_pc=1）')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--use_ros2', action='store_true', help='是否从 ROS2 话题读取图像而不是从文件读取')
  parser.add_argument('--left_topic', type=str,
                      default='/image_left_raw/nv12_quarter_hengfortwoCamera2depthimage',
                      help='左目图像话题名 (NV12)')
  parser.add_argument('--right_topic', type=str,
                      default='/image_right_raw/nv12_quarter_hengfortwoCamera2depthimage',
                      help='右目图像话题名 (NV12)')
  parser.add_argument('--ros2_timeout', type=float, default=5.0,
                      help='等待一对左右图像的超时时间 (秒)')
  parser.add_argument('--no_show_pc', action='store_true', help='save_pc=1 时不弹出 Open3D 点云窗口，仅保存 ply 文件')
  parser.add_argument('--show_vis', type=int, default=1, help='为 1 时在推理结束后用窗口显示 vis.png')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir, weights_only=False)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  img0, img1 = get_input_images(args)
  # 如果是带 alpha 通道的 RGBA 图像，裁掉 alpha，只保留前 3 个通道 (RGB)
  if img0.ndim == 3 and img0.shape[2] == 4:
    img0 = img0[..., :3]
  if img1.ndim == 3 and img1.shape[2] == 4:
    img1 = img1[..., :3]
  scale = args.scale
  assert scale<=1, "scale must be <=1"
  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  logging.info(f"img0: {img0.shape}")

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  # 只统计一次前向推理（从输入张量到视差）的时间
  # CUDA 是异步的，这里用 synchronize() 确保计时精准
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  start_time = time.time()
  with torch.cuda.amp.autocast(True):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  infer_time = time.time() - start_time
  # 同时用 logging 和 print 打印，用明显的分隔符方便在终端中快速看到
  logging.info(f"Inference time (stereo depth only): {infer_time:.3f} s")
  print("\n" + "="*70)
  print(f"[FoundationStereo] Inference time (stereo depth only): {infer_time:.3f} s")
  print("="*70 + "\n")
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H,W)
  vis = vis_disparity(disp)
  vis = np.concatenate([img0_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/vis.png', vis)
  logging.info(f"Output saved to {args.out_dir}")

  if args.show_vis:
    try:
      cv2.imshow("FoundationStereo vis", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    except Exception as e:
      logging.warning(f"Failed to show vis window: {e}")

  if args.remove_invisible:
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

  if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    logging.info(f"depth_meter.npy saved to {args.out_dir}")

    if args.save_pc:
      xyz_map = depth2xyzmap(depth, K)
      pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
      keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
      keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
      pcd = pcd.select_by_index(keep_ids)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
      logging.info(f"PCL saved to {args.out_dir}")

      if args.denoise_cloud:
        logging.info("[Optional step] denoise point cloud...")
        cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
        inlier_cloud = pcd.select_by_index(ind)
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
        pcd = inlier_cloud

      if not args.no_show_pc:
        logging.info("Visualizing point cloud. Press ESC to exit.")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
        vis.run()
        vis.destroy_window()
      else:
        logging.info("已关闭 Open3D 点云显示，点云已保存至 %s", args.out_dir)

