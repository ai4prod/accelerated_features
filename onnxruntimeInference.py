import numpy as np
import os
import onnxruntime as ort
import tqdm
import cv2
import matplotlib.pyplot as plt

#TO RUN THIS CODE python3.8 onnxruntimeInference.py because the tensorrt backend is only supported in python3.8


model_path = 'xfeat_matching_tensorrt.onnx'    # python ./export.py --dynamic --export_path ./xfeat_matching.onnx

#Load some example images
im1 = cv2.imread('assets/Refernce_512_0000.png', cv2.IMREAD_COLOR)
im2 = cv2.imread('assets/Centred_512_0002.png', cv2.IMREAD_COLOR)

im3 = cv2.imread('assets/Refernce_512_0000.png', cv2.IMREAD_COLOR)
im4 = cv2.imread('assets/Centred_512_0002.png', cv2.IMREAD_COLOR)

print(f"im1 shape: {im1.shape}")
print(f"im2 shape: {im2.shape}")
print(f"im3 shape: {im3.shape}")
print(f"im4 shape: {im4.shape}")



def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    print("h, w", h, w)
    
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    print("matches", len(matches))
    input("t")
    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)
    #print image shape
    print(img_matches.shape)


    return img_matches


tmp_ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

providers = [
    #The TensorrtExecutionProvider is the fastest.
    ('TensorrtExecutionProvider', { 
        'device_id': 0,
        'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_engine_cache',
        'trt_engine_cache_prefix': 'model',
        'trt_dump_subgraphs': False,
        'trt_timing_cache_enable': True,
        'trt_timing_cache_path': './trt_engine_cache',
        #'trt_builder_optimization_level': 3,
    }),

    # The CUDAExecutionProvider is slower than PyTorch, 
    # possibly due to performance issues with large matrix multiplication "cossim = torch.bmm(feats1, feats2.permute(0,2,1))"
    # Reducing the top_k value when exporting to ONNX can decrease the matrix size.
    # ('CUDAExecutionProvider', { 
    #     'device_id': 0,
    #     'gpu_mem_limit': 12 * 1024 * 1024 * 1024,
    # }),
    # ('CPUExecutionProvider',{ 
    # })
    ]
ort_session = ort.InferenceSession(model_path, providers=providers)


#PREPARE INPUTS

im1 = cv2.resize(im1, (480, 480))
im2= cv2.resize(im2, (480, 480))

im3= cv2.resize(im3, (480, 480))
im4= cv2.resize(im4, (480, 480))

print(f"im1 shape: {im1.shape}")
print(f"im2 shape: {im2.shape}")
print(f"im3 shape: {im3.shape}")
print(f"im4 shape: {im4.shape}")

input_array_1 = im1.transpose(2, 0, 1).astype(np.float32)
input_array_1 = np.expand_dims(input_array_1, axis=0)
input_array_2 = im2.transpose(2, 0, 1).astype(np.float32)
input_array_2 = np.expand_dims(input_array_2, axis=0)


input_array_3 = im3.transpose(2, 0, 1).astype(np.float32)
input_array_3 = np.expand_dims(input_array_3, axis=0)
input_array_4 = im4.transpose(2, 0, 1).astype(np.float32)
input_array_4 = np.expand_dims(input_array_4, axis=0)

batch_size =2

print(input_array_1[0])

input("t")

# Psuedo-batch the input images
# input_array_1 = np.concatenate([input_array_1 for _ in range(batch_size)], axis=0)
# input_array_2 = np.concatenate([input_array_2 for _ in range(batch_size)], axis=0)

print(f"input_array_1 shape: {input_array_1.shape}")
print(f"input_array_2 shape: {input_array_2.shape}")
print(f"input_array_3 shape: {input_array_3.shape}")
print(f"input_array_4 shape: {input_array_4.shape}")

input_array_1 = np.concatenate([input_array_1,input_array_3], axis=0)
input_array_2 = np.concatenate([input_array_2,input_array_4], axis=0)

print(f"input_array_1 shape: {input_array_1.shape}")

input("t")

inputs = {
    ort_session.get_inputs()[0].name: input_array_1,
    ort_session.get_inputs()[1].name: input_array_2
}


#RUN MATCHING

outputs = ort_session.run(None, inputs)

# Validate the outputs of the psuedo-batched inputs
matches = outputs[0]

print(f"matches shape: {matches.shape}")


batch_indexes = outputs[1]
print(batch_indexes)

print(f"batch_indexes shape: {batch_indexes.shape}")
input("t")

matches_0 = matches[batch_indexes == 0]
valid = []
for i in range(1, input_array_1.shape[0]):
    valid.append(np.all(matches_0 == matches[batch_indexes == i]))
print(f"equal: {valid}")



import time

# Run the model 100 times to get an average time
times = []
for i in tqdm.tqdm(range(100)):
    start = time.time()
    outputs = ort_session.run(None, inputs)
    times.append(time.time() - start)

print(f"Average time per batch: {np.mean(times):.4f} seconds")
print(f"Average time per image: {np.mean(times)/batch_size:.4f} seconds")
print(f"Average FPS per image: {batch_size/np.mean(times):.4f}")


matches = outputs[0]
batch_indexes = outputs[1]

print(batch_indexes)
input("t")
mkpts_0, mkpts_1 = matches[batch_indexes == 0][..., :2], matches[batch_indexes == 0][..., 2:]

canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
plt.figure(figsize=(12,12))
plt.imshow(canvas[..., ::-1])

# save the image
plt.savefig('matches_small.png')