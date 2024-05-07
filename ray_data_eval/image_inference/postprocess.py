from ray_data_pipeline_helpers import (
    postprocess,
)

BATCH_SIZE = 128
DATA_PERCENTAGE = 100
COMPUTE = "1+1"

OUTPUT_FILENAME = f"{COMPUTE}_image_inference_s3_batch_{BATCH_SIZE}_{DATA_PERCENTAGE}pct.out"


def main():
    postprocess(OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
