import "./style.css";
import * as posenet_module from "@tensorflow-models/posenet";
import * as tf from "@tensorflow/tfjs";
import {
  drawKeypoints,
  drawPoint,
  drawSkeleton,
  isMobile,
  toggleLoadingUI,
  setStatusText,
  drawCircle,
  checkCollision,
} from "./utils/utils";
import { Circle } from "./utils/circle";

// Camera stream video element
let video;
let videoWidth = 1000;
let videoHeight = 1000;

// Canvas
let canvasWidth = 800;
let canvasHeight = 800;

// ML models
let facemesh;
let posenet;
let minPoseConfidence = 0.15;
let minPartConfidence = 0.1;
let nmsRadius = 30.0;

// Misc
let mobile = false;
const defaultPoseNetArchitecture = "MobileNetV1";
const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 200;

let circle = new Circle(300, 300, 50, "black");
let rightWrist;

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Browser API navigator.mediaDevices.getUserMedia not available"
    );
  }

  const video = document.getElementById("video");
  video.width = videoWidth;
  video.height = videoHeight;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: videoWidth,
      height: videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video) {
  const canvas = document.getElementById("output");
  const keypointCanvas = document.getElementById("keypoints");
  const videoCtx = canvas.getContext("2d");
  const keypointCtx = keypointCanvas.getContext("2d");

  canvas.width = videoWidth;
  canvas.height = videoHeight;
  keypointCanvas.width = videoWidth;
  keypointCanvas.height = videoHeight;

  async function poseDetectionFrame() {
    // Begin monitoring code for frames per second
    let poses = [];

    drawCircle(keypointCtx, circle);

    // Creates a tensor from an image
    const input = tf.browser.fromPixels(canvas);
    let all_poses = await posenet.estimatePoses(video, {
      flipHorizontal: true,
      decodingMethod: "multi-person",
      maxDetections: 1,
      scoreThreshold: minPartConfidence,
      nmsRadius: nmsRadius,
    });

    poses = poses.concat(all_poses);
    input.dispose();

    keypointCtx.clearRect(0, 0, videoWidth, videoHeight);
    poses.forEach(({ score, keypoints }) => {
      // console.log(keypoints[10]);
      if (score >= minPoseConfidence) {
        drawKeypoints(keypoints, minPartConfidence, keypointCtx);
        drawSkeleton(keypoints, minPartConfidence, keypointCtx);
      }
      rightWrist = new Circle(
        keypoints[10].position.x,
        keypoints[10].position.y,
        1,
        "rightWrist"
      );
      if (checkCollision(rightWrist, circle)) {
        // console.log("true");
        circle.name = "green";
      } else {
        circle.name = "black";
      }
    });

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

function setupCanvas() {
  mobile = isMobile();
  if (mobile) {
    canvasWidth = Math.min(window.innerWidth, window.innerHeight);
    canvasHeight = canvasWidth;
    videoWidth *= 0.7;
    videoHeight *= 0.7;
  }
}

export async function bindPage() {
  setupCanvas();

  posenet = await posenet_module.load({
    architecture: defaultPoseNetArchitecture,
    outputStride: defaultStride,
    inputResolution: defaultInputResolution,
    multiplier: defaultMultiplier,
    quantBytes: defaultQuantBytes,
  });

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById("info");
    info.textContent =
      "this device type is not supported yet, " +
      "or this browser does not support video capture: " +
      e.toString();
    info.style.display = "block";
    throw e;
  }

  detectPoseInRealTime(video, posenet);
}

bindPage();
