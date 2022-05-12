# Camo Worms

A project for CITS4404


## Running

### In Docker

*These instructions assume that you have docker installed.*

The project can be run in docker to improve portability.

**Note:** files will be created in the `progress` directory.
These files will be owned by **root**.

```bash
chmod +x ./bin/* # Update script permissions to be runable

# Builds the docker image
# This needs to be rebuilt after every code change
./bin/build

# Start a container based on the created image
# The number of generations can be controlled and defaults to 10
./bin/start <number of generations>
```
