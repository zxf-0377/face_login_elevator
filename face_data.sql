/*
 Navicat Premium Data Transfer

 Source Server         : conn1
 Source Server Type    : MySQL
 Source Server Version : 50562
 Source Host           : localhost:3306
 Source Schema         : face_data

 Target Server Type    : MySQL
 Target Server Version : 50562
 File Encoding         : 65001

 Date: 02/02/2021 10:46:56
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for face_json
-- ----------------------------
DROP TABLE IF EXISTS `face_json`;
CREATE TABLE `face_json`  (
  `id` int(32) NOT NULL AUTO_INCREMENT COMMENT 'id自增',
  `ugroup` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '用户群组',
  `uid` varchar(64) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '图片用户id',
  `json` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '人脸的向量',
  `pic_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '图片名称',
  `date` datetime NULL DEFAULT NULL COMMENT '插入时间',
  `state` tinyint(1) NULL DEFAULT NULL,
  `floor` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '楼层',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 51 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

SET FOREIGN_KEY_CHECKS = 1;
