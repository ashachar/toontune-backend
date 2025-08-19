SAM2 Video API:

import replicate

input = {
    "mask_type": "highlighted",
    "video_fps": 25,
    "input_video": "https://api.replicate.com/v1/files/N2E3N2I3NTgtNzkxMS00ZDU3LWJkNWMtMWQxMDQ5NjEyNjZh/download?expiry=1723645268&owner=pwntus&signature=w0ZTXY7u4SnrIJYRyuA86uP81JJ46VjSJYCrxAY%252Fwwc%253D",
    "click_frames": "1",
    "output_video": True,
    "click_object_ids": "bee_1,bee_2,bee_3,bee_4,bee_5,bee_6,bee_7,bee_8",
    "click_coordinates": "[391,239],[178,320],[334,391],[185,446],[100,433],[461,499],[11,395],[9,461]"
}

for event in replicate.stream(
    "meta/sam-2-video:33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d",
    input=input
):
    print(event)
    #=> "https://replicate.delivery/pbxt/iGyFurounuZkAFqPL7Rjq5bzL9WdE7AhUfdXmKlvEnroHTpJA/output_video.mp4"


---


SAM2 Image API:

import replicate

input = {
    "image": "https://replicate.delivery/pbxt/LMbGi83qiV3QXR9fqDIzTl0P23ZWU560z1nVDtgl0paCcyYs/cars.jpg"
}

output = replicate.run(
    "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
    input=input
)

print(output)
#=> {"combined_mask":"https://replicate.delivery/pbxt/PhfVJub...