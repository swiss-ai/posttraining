mkdir -p /iopsstor/scratch/cscs/$(id -un)/projects/posttraining/artifacts/private

if [ -z "$INSTANCE" ]; then
  echo "Please set the INSTANCE variable to the desired instance name (dev or run)."
  exit 1
fi

rm -f /users/$(id -un)/projects/posttraining/$INSTANCE/artifacts/private
rm -f /users/$(id -un)/projects/posttraining/$INSTANCE/artifacts/shared
rm -f /users/$(id -un)/projects/posttraining/$INSTANCE/artifacts/store
ln -s /iopsstor/scratch/cscs/$(id -un)/projects/posttraining/artifacts/private /users/$(id -un)/projects/posttraining/$INSTANCE/artifacts/private
ln -s /iopsstor/scratch/cscs/smoalla/projects/posttraining/artifacts/shared /users/$(id -un)/projects/posttraining/$INSTANCE/artifacts/shared
ln -s /capstor/store/cscs/swissai/infra01/posttraining/artifacts /users/$(id -un)/projects/posttraining/$INSTANCE/artifacts/store
