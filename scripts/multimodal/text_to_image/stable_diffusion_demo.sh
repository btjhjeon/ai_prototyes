num_samples=10

# base_model='SD_1_5'
# base_model='SD_2_1'
# base_model='merged_0.5_1.0'
# base_model='merged_0.25+0.25+0.5_1.0'
# base_model="threemun_1000_1.0"
base_model="RV_1.4"
# for base_model in "SD_1_5" "threemun_1000_1.0" "merged_0.5_1.0"
# do

# prompt='Cheongsudang cafe, 31-9 Donhwamun-ro 11na-gil, Ikseon-dong, Jongno-gu, Seoul, South Korea'
# prompt='Gwanghwamun, 172, Sejong-daero, Jongno-gu, Seoul'
# source_name='starbucks_kj'
# prompt='inside the Gyeongju-Daereungwon branch Starbucks cafe, 125, Cheomseong-ro, Gyeongju-si, Gyeongsangbuk-do, Republic of Korea'
prompt="a photo of <s1>"
# prompt="a photo of <s1>, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
# neg_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, disfigured, gross proportions"

# iteration=2000
# for iteration in 1000 2000 3000
for iteration in 500 1000 2000 # 3000
do

# source_name='cheongsudang'
# source_name="changuimun"
# source_name="dongdaemun"
source_name="sungnyemun"
# source_name="threemun"
# source_name="gwanghwamun"
# for source_name in "changuimun" "dongdaemun" "gwanghwamun" "sungnyemun" "threemun"
# do

# name="ch+do+gw"
# for gamma in 0.3 0.4 0.5
# do

# source_name="${name}_${gamma}"
# source_name="ch+do+gw_0.25+0.25+0.5"

# lora_alpha=1.0
# for lora_alpha in 0.5 0.7 1.0 1.2
# do


# model_dir='runwayml/'
# model_name='stable-diffusion-v1-5'
# model_dir='stabilityai/'
# model_name='stable-diffusion-2-1'
model_dir='/t2meta/dataset/stable-diffusion-models/custom_models'
model_name="${base_model}__${source_name}__dreambooth_${iteration}"
# model_name="${base_model}__${source_name}__LoRA_${iteration}_${lora_alpha}"
# model_name='SD_1_5__'${source_name}'__1000'
# model_name='SD_1_5__'${source_name}'__2000'
# model_name='SD_1_5__'${source_name}'__3000'
# model_name='SD_1_5__'${source_name}'__LoRA_1000'
# model_name='SD_1_5__'${source_name}'__LoRA_2000'
# model_name='SD_2_1__'${source_name}'__1000'
# model_name='SD_2_1__'${source_name}'__2000'
# model_name='SD_2_1__'${source_name}'__LoRA_2000'

model_path="${model_dir}/${model_name}"
output_dir="${model_path}/results"
output_path="${output_dir}/${prompt}.jpg"

python -m scripts.text_to_image.test_stable_diffusion \
    -m=${model_path} \
    -p="${prompt}" \
    --num_samples=${num_samples} \
    -o="${output_path}"

python -m scripts.merge_images \
    -i \
    "${output_dir}/${prompt}_000.jpg" \
    "${output_dir}/${prompt}_001.jpg" \
    "${output_dir}/${prompt}_002.jpg" \
    "${output_dir}/${prompt}_003.jpg" \
    --rows=2 \
    -o="${output_path}"

done
# done
# done
# done
