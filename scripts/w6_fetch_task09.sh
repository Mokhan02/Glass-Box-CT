#!/bin/bash
# W6: MSD Task09 (spleen; CC-BY-SA 4.0) for the forgetting guard, from the same anonymous S3 source as Task03 (W1). Verifies the size against the object, `tar -tf`, records the SHA-256, extracts into its own empty directory.
W=/lambda/nfs/CTGate/w6; D=$W/data; URL=https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar; mkdir -p $D; cd $D || exit 2
[ -d Task09_Spleen/imagesTr ] && { echo "TASK09_PRESENT"; exit 0; }
SIZE=$(curl -sI $URL | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}'); echo "source size $SIZE"
curl -s -o Task09_Spleen.tar $URL || { echo "TASK09_DOWNLOAD_FAILED"; exit 1; }
[ "$(stat -c %s Task09_Spleen.tar)" = "$SIZE" ] || { echo "TASK09_SIZE_MISMATCH"; exit 1; }
tar -tf Task09_Spleen.tar > /dev/null || { echo "TASK09_TAR_BAD"; exit 1; }
sha256sum Task09_Spleen.tar | tee $D/Task09_Spleen.tar.sha256; tar -xf Task09_Spleen.tar --exclude='._*' || exit 1
ls Task09_Spleen/imagesTr | grep -v '^\._' | wc -l | xargs echo "TASK09_IMAGES"; echo TASK09_OK
