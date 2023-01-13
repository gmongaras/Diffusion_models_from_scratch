import zipfile
import os
import pickle




# Create the directory for images and classes
if not os.path.exists("Imagenet64"):
    os.mkdir("Imagenet64")

# index counter
idx_ctr = 0

# Unique class set
unique_cls = set()

# Read the pickle data
for archive in ["Imagenet64_train_part1.zip", "Imagenet64_train_part2.zip"]:
    # Load the archive
    with zipfile.ZipFile(archive, 'r') as archive_ld:

        # Iterate over all archive data
        for filename in archive_ld.filelist:
            # Load the file data
            file = pickle.load(archive_ld.open(filename.filename, "r"))

            # Iterate over all image,label pairs
            for (img, label) in zip(file["data"], file["labels"]):

                # Create a new dictionary for this data
                img_label = dict(
                    img=img,
                    label=label
                )

                # Save the label to the uniue class set
                unique_cls.add(label)

                # Save the img,label as a separate file
                with open(f"Imagenet64/{idx_ctr}.pkl", "wb") as f:
                    pickle.dump(img_label, f)

                # Increase the counter
                idx_ctr += 1
            
            del file


# Save metadata about the number of data and
# number of classes in the data
with open(f"Imagenet64/metadata.pkl", "wb") as f:
    pickle.dump(dict(
        num_data=idx_ctr,
        cls_min=min(unique_cls),
        cls_max=max(unique_cls)
    ), f)