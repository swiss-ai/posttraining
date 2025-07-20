mkdir -p /iopsstor/scratch/cscs/$(id -un)/projects/swiss-alignment/artifacts/private

if [ -z "$INSTANCE" ]; then
  echo "Please set the INSTANCE variable to the desired instance name (dev or run)."
  exit 1
fi

rm -f /users/$(id -un)/projects/swiss-alignment/$INSTANCE/artifacts/private
rm -f /users/$(id -un)/projects/swiss-alignment/$INSTANCE/artifacts/shared
rm -f /users/$(id -un)/projects/swiss-alignment/$INSTANCE/artifacts/store
ln -s /iopsstor/scratch/cscs/$(id -un)/projects/swiss-alignment/artifacts/private /users/$(id -un)/projects/swiss-alignment/$INSTANCE/artifacts/private
ln -s /iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared /users/$(id -un)/projects/swiss-alignment/$INSTANCE/artifacts/shared
ln -s /capstor/store/cscs/swissai/infra01/swiss-alignment/artifacts /users/$(id -un)/projects/swiss-alignment/$INSTANCE/artifacts/store
