# Multi-FPGA in FINN+
Multi-FPGA usage in FINN+ is implemented purely in the second half of the FINN-flow. This means you can use FINN as normal, and even use the same steps - Multi-FPGA is implicitly switched on by providing a partitioning configuration in your flow config file. As soon as these settings are detected, FINN+ in the background switches the second half of the flow to the Multi-FPGA specific steps and transformations.

## How to use
In your configuration file, simply provide a value in the `partitioning_configuration` field. Most fields have sensible defaults, but at least `num_fpgas` and `parallel_synthesis_workers` should be set manually.
The flow will automatically switch to MultiFPGA. The difference should not be noticeable - only some more logging output. If everything worked correctly, instead of one `finn-accel.xclbin`, you should see `finn-accel-0.xclbin, finn-accel-1.xclbin` and so on.

An example configuration could look like this:
```YAML
partitioning_configuration:
  num_fpgas: 2
  partition_strategy: resource_utilization
  topology: chain
  communication_kernel: aurora
  max_utilization: 0.8
  parallel_synthesis_workers: 2
```

Such a configuration would try to partition the design onto 2 devices. To let the partitioner decide itself (based on estimate reports), select `-1`. The used partition strategy is `resource_utilization`, meaning that a balanced resource utilization is used as the metric to judge a good solution. The `chain` topology constrains the solver to a certain assignment of device IDs. `aurora` specifies which communication technology to use. `max_utilization` is used by the solver to make sure that the FPGAs resources are not overutilized. Finally, `parallel_synthesis_workers` specifies how many synthesis processes can run in parallel. 

Now simply run FINN+ as usual: `finn build cfg.yaml`.

## Multi-FPGA specific steps
There are 3-4 Multi-FPGA specific steps that need to be executed before synthesis. Almost all of them are done in the `step_prepare_synthesis` dataflow step, which should be run after FIFO sizing and IPGen but before the actual synthesis.

### 1. Partitioning
To decide which layers/nodes are put onto which device, partitioning is done. If an existing assignment exists, this can be passed as an argument. If no assignment exists yet, the dataflow graph is converted into a (M)-ILP model. This model is then solved to assign every node to one device ID.

#### Technical Details
`partitioner.py` implements the `Partitioner` abstract baseclass. When constructed, the object resolves to the correct solver and performs some specific fixes. It also provides abstract basemethods for specific partitioners to implement.
A specific partitioner should check whether a partitioning is feasible, implement the solving methods and the construction of the actual model.

Utilizing the partitioners is done using `ApplyPartitioning` and `PartitionForMultiFPGA`, both of which are found in `partition_model.py`. The former recieves either a mapping of layer names to device IDs or a path to a YAML file containing such mappings. These mappings are then applied and the `device_id` attribute of all nodes is set.

`PartitionForMultiFPGA` calls on the partitioners to solve the model. If no specific device-count was given, it will automatically try to find a good estimate until a valid partitioning configuration is found. It will then write the result into the reports directory and call `ApplyPartitioning` to apply the solution to the graph.


### 2. Multi-FPGA StreamingDataflowPartitions
As soon as the assignment of the device IDs is complete, consecutive nodes with the same device ID are grouped together into _StreamingDataflowPartition_ nodes (_SDPs_). These are meta-nodes that act as containers for subgraphs. The SDP is then assigned the same device ID as all of the nodes it contains.

#### Technical Details
First, `ClusterByNodeAttribute` in `create_multi_sdp.py` will cluster nodes on the attribute `partition_id` by the attribute `device_id`. This is not exactly equivalent to simply setting `partition_id == device_id`, since this way gaps in the graph are kept, and in a chain of devices `1-2-1`, there are actually two SDPs created for device `1`.

Since this can lead to cycles in the graph, these need to be resolved. Afterwards, the actual SDPs are being created through `CreateDataflowPartition`.

### 3. Metadata Creation
At this point, an internal model of the finished accelerator design is created. It stores which SDPs are connected with which other SDPs on which devices, where the communication kernels are placed, how many ports each design uses, if a port is used for TX, RX or both, etc. This metadata is saved alongside the ONNX model.

#### Technical Details
Since different communication methods may work slightly differently, each communication kernel also has a matching class inheriting from `NetworkMetadata`. The baseclass specifies some elementary methods which need to be implemented. The transformation `CreateNetworkMetadata` reads the model and automatically creates the correct type of metadata. Reading the model and creating the metadata is always done in the same way, regardless of communication kernel - hence only the base `NetworkMetadata` methods are used. The path to this data is then stored in the metadata property `network_metadata` and can at any point be read again.

### 4. Communication Kernel Preparation
At this point, depending on which communication methodology is used, custom preparation steps can be done. For example, the _AuroraFlow_ kernel needs to be configured and packaged into an XO file to be used at the linking stage. Such preparations can be done in this step (but are not required).

#### Technical Details
`PrepareCommunicationKernels` checks which communication kernel is used and runs the matching preparation transformations.

### (Synthesis)
After everything is done, the SDPs are packaged into XOs as well (Vitis flow only) and the linker configuration is created. For Multi-FPGA, an additional transformation will be executed. This additional transformation modifies the linker configuration to, for example, instantiate the communication kernel and connect its stream interface with the compute kernel.
