
Ļ#Ø#
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
ø
AsString

input"T

output"
Ttype:
2		
"
	precisionint’’’’’’’’’"

scientificbool( "
shortestbool( "
widthint’’’’’’’’’"
fillstring 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ļ
ParseExample

serialized	
names
sparse_keys*Nsparse

dense_keys*Ndense
dense_defaults2Tdense
sparse_indices	*Nsparse
sparse_values2sparse_types
sparse_shapes	*Nsparse
dense_values2Tdense"
Nsparseint("
Ndenseint("%
sparse_types
list(type)(:
2	"
Tdense
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.1-dev201904122unknown£¹

global_step/Initializer/zerosConst*
value	B	 R *
_output_shapes
: *
dtype0	*
_class
loc:@global_step

global_step
VariableV2*
shared_name *
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
	container 
²
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_output_shapes
: *
_class
loc:@global_step
o
input_example_tensorPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
U
ParseExample/ConstConst*
valueB *
_output_shapes
: *
dtype0
W
ParseExample/Const_1Const*
valueB *
_output_shapes
: *
dtype0
W
ParseExample/Const_2Const*
valueB *
_output_shapes
: *
dtype0
W
ParseExample/Const_3Const*
valueB *
_output_shapes
: *
dtype0
W
ParseExample/Const_4Const*
valueB *
_output_shapes
: *
dtype0
W
ParseExample/Const_5Const*
valueB *
_output_shapes
: *
dtype0
W
ParseExample/Const_6Const*
valueB *
_output_shapes
: *
dtype0
b
ParseExample/ParseExample/namesConst*
valueB *
_output_shapes
: *
dtype0
h
&ParseExample/ParseExample/dense_keys_0Const*
value	B Ba*
_output_shapes
: *
dtype0
h
&ParseExample/ParseExample/dense_keys_1Const*
value	B Bb*
_output_shapes
: *
dtype0
h
&ParseExample/ParseExample/dense_keys_2Const*
value	B Bc*
_output_shapes
: *
dtype0
h
&ParseExample/ParseExample/dense_keys_3Const*
value	B Bd*
_output_shapes
: *
dtype0
h
&ParseExample/ParseExample/dense_keys_4Const*
value	B Be*
_output_shapes
: *
dtype0
h
&ParseExample/ParseExample/dense_keys_5Const*
value	B Bf*
_output_shapes
: *
dtype0
h
&ParseExample/ParseExample/dense_keys_6Const*
value	B Bg*
_output_shapes
: *
dtype0
“
ParseExample/ParseExampleParseExampleinput_example_tensorParseExample/ParseExample/names&ParseExample/ParseExample/dense_keys_0&ParseExample/ParseExample/dense_keys_1&ParseExample/ParseExample/dense_keys_2&ParseExample/ParseExample/dense_keys_3&ParseExample/ParseExample/dense_keys_4&ParseExample/ParseExample/dense_keys_5&ParseExample/ParseExample/dense_keys_6ParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4ParseExample/Const_5ParseExample/Const_6*
sparse_types
 *<
dense_shapes,
*:::::::*
Ndense*
_output_shapes
:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’*
Nsparse *
Tdense
	2

2dnn/input_from_feature_columns/input_layer/a/ShapeShapeParseExample/ParseExample*
out_type0*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/a/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/a/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/a/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ś
:dnn/input_from_feature_columns/input_layer/a/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/a/Shape@dnn/input_from_feature_columns/input_layer/a/strided_slice/stackBdnn/input_from_feature_columns/input_layer/a/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/a/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
~
<dnn/input_from_feature_columns/input_layer/a/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
ö
:dnn/input_from_feature_columns/input_layer/a/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/a/strided_slice<dnn/input_from_feature_columns/input_layer/a/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:
Ö
4dnn/input_from_feature_columns/input_layer/a/ReshapeReshapeParseExample/ParseExample:dnn/input_from_feature_columns/input_layer/a/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

2dnn/input_from_feature_columns/input_layer/b/ShapeShapeParseExample/ParseExample:1*
out_type0*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/b/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/b/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/b/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ś
:dnn/input_from_feature_columns/input_layer/b/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/b/Shape@dnn/input_from_feature_columns/input_layer/b/strided_slice/stackBdnn/input_from_feature_columns/input_layer/b/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/b/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
~
<dnn/input_from_feature_columns/input_layer/b/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
ö
:dnn/input_from_feature_columns/input_layer/b/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/b/strided_slice<dnn/input_from_feature_columns/input_layer/b/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:
Ų
4dnn/input_from_feature_columns/input_layer/b/ReshapeReshapeParseExample/ParseExample:1:dnn/input_from_feature_columns/input_layer/b/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

2dnn/input_from_feature_columns/input_layer/c/ShapeShapeParseExample/ParseExample:2*
out_type0*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/c/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/c/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/c/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ś
:dnn/input_from_feature_columns/input_layer/c/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/c/Shape@dnn/input_from_feature_columns/input_layer/c/strided_slice/stackBdnn/input_from_feature_columns/input_layer/c/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/c/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
~
<dnn/input_from_feature_columns/input_layer/c/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
ö
:dnn/input_from_feature_columns/input_layer/c/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/c/strided_slice<dnn/input_from_feature_columns/input_layer/c/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:
Ų
4dnn/input_from_feature_columns/input_layer/c/ReshapeReshapeParseExample/ParseExample:2:dnn/input_from_feature_columns/input_layer/c/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

2dnn/input_from_feature_columns/input_layer/d/ShapeShapeParseExample/ParseExample:3*
out_type0*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/d/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/d/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/d/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ś
:dnn/input_from_feature_columns/input_layer/d/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/d/Shape@dnn/input_from_feature_columns/input_layer/d/strided_slice/stackBdnn/input_from_feature_columns/input_layer/d/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/d/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
~
<dnn/input_from_feature_columns/input_layer/d/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
ö
:dnn/input_from_feature_columns/input_layer/d/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/d/strided_slice<dnn/input_from_feature_columns/input_layer/d/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:
Ų
4dnn/input_from_feature_columns/input_layer/d/ReshapeReshapeParseExample/ParseExample:3:dnn/input_from_feature_columns/input_layer/d/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

2dnn/input_from_feature_columns/input_layer/e/ShapeShapeParseExample/ParseExample:4*
out_type0*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/e/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/e/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/e/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ś
:dnn/input_from_feature_columns/input_layer/e/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/e/Shape@dnn/input_from_feature_columns/input_layer/e/strided_slice/stackBdnn/input_from_feature_columns/input_layer/e/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/e/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
~
<dnn/input_from_feature_columns/input_layer/e/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
ö
:dnn/input_from_feature_columns/input_layer/e/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/e/strided_slice<dnn/input_from_feature_columns/input_layer/e/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:
Ų
4dnn/input_from_feature_columns/input_layer/e/ReshapeReshapeParseExample/ParseExample:4:dnn/input_from_feature_columns/input_layer/e/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

2dnn/input_from_feature_columns/input_layer/f/ShapeShapeParseExample/ParseExample:5*
out_type0*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/f/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/f/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/f/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ś
:dnn/input_from_feature_columns/input_layer/f/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/f/Shape@dnn/input_from_feature_columns/input_layer/f/strided_slice/stackBdnn/input_from_feature_columns/input_layer/f/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/f/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
~
<dnn/input_from_feature_columns/input_layer/f/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
ö
:dnn/input_from_feature_columns/input_layer/f/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/f/strided_slice<dnn/input_from_feature_columns/input_layer/f/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:
Ų
4dnn/input_from_feature_columns/input_layer/f/ReshapeReshapeParseExample/ParseExample:5:dnn/input_from_feature_columns/input_layer/f/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

2dnn/input_from_feature_columns/input_layer/g/ShapeShapeParseExample/ParseExample:6*
out_type0*
T0*
_output_shapes
:

@dnn/input_from_feature_columns/input_layer/g/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/g/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0

Bdnn/input_from_feature_columns/input_layer/g/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
Ś
:dnn/input_from_feature_columns/input_layer/g/strided_sliceStridedSlice2dnn/input_from_feature_columns/input_layer/g/Shape@dnn/input_from_feature_columns/input_layer/g/strided_slice/stackBdnn/input_from_feature_columns/input_layer/g/strided_slice/stack_1Bdnn/input_from_feature_columns/input_layer/g/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
~
<dnn/input_from_feature_columns/input_layer/g/Reshape/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
ö
:dnn/input_from_feature_columns/input_layer/g/Reshape/shapePack:dnn/input_from_feature_columns/input_layer/g/strided_slice<dnn/input_from_feature_columns/input_layer/g/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:
Ų
4dnn/input_from_feature_columns/input_layer/g/ReshapeReshapeParseExample/ParseExample:6:dnn/input_from_feature_columns/input_layer/g/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

6dnn/input_from_feature_columns/input_layer/concat/axisConst*
valueB :
’’’’’’’’’*
_output_shapes
: *
dtype0
¶
1dnn/input_from_feature_columns/input_layer/concatConcatV24dnn/input_from_feature_columns/input_layer/a/Reshape4dnn/input_from_feature_columns/input_layer/b/Reshape4dnn/input_from_feature_columns/input_layer/c/Reshape4dnn/input_from_feature_columns/input_layer/d/Reshape4dnn/input_from_feature_columns/input_layer/e/Reshape4dnn/input_from_feature_columns/input_layer/f/Reshape4dnn/input_from_feature_columns/input_layer/g/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*'
_output_shapes
:’’’’’’’’’*

Tidx0
Å
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"   
   *
_output_shapes
:*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
·
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *0æ*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
·
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *0?*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
seed2 *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
T0*
_output_shapes

:
*
dtype0*

seed 

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
¬
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*
_output_shapes

:
*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0

:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes

:
*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
ß
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*0
shared_name!dnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
dtype0*
shape
:
*
	container 

@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
Ų
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
Ē
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
®
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB
*    *
_output_shapes
:
*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
Õ
dnn/hiddenlayer_0/bias/part_0VarHandleOp*.
shared_namednn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
: *
dtype0*
shape:
*
	container 

>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
Ē
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0
½
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0

'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
dtype0
v
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
T0*
_output_shapes

:

Ē
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:’’’’’’’’’


%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
*
dtype0
n
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
T0*
_output_shapes
:


dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’

k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’

g
dnn/zero_fraction/SizeSizednn/hiddenlayer_0/Relu*
out_type0	*
T0*
_output_shapes
: 
c
dnn/zero_fraction/LessEqual/yConst*
valueB	 R’’’’*
_output_shapes
: *
dtype0	

dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction/cond/SwitchSwitchdnn/zero_fraction/LessEqualdnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
m
dnn/zero_fraction/cond/switch_tIdentitydnn/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
k
dnn/zero_fraction/cond/switch_fIdentitydnn/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
h
dnn/zero_fraction/cond/pred_idIdentitydnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: 

*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
Ļ
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*
T0*'
_output_shapes
:’’’’’’’’’

ę
4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*:
_output_shapes(
&:’’’’’’’’’
:’’’’’’’’’
*)
_class
loc:@dnn/hiddenlayer_0/Relu
±
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*'
_output_shapes
:’’’’’’’’’
*
Truncate( *

SrcT0


*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
valueB"       *
_output_shapes
:*
dtype0
Ī
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0

dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*
_output_shapes
: *
Truncate( *

SrcT0

,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
valueB
 *    *
_output_shapes
: *
dtype0
Ó
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:’’’’’’’’’

č
6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*:
_output_shapes(
&:’’’’’’’’’
:’’’’’’’’’
*)
_class
loc:@dnn/hiddenlayer_0/Relu
µ
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*'
_output_shapes
:’’’’’’’’’
*
Truncate( *

SrcT0


,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
valueB"       *
_output_shapes
:*
dtype0
Ō
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
	keep_dims( *
T0	*
_output_shapes
: *

Tidx0
¤
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 

(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 

)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0	

+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0	
°
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
_output_shapes
: *
dtype0
Æ
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
_output_shapes
: *
dtype0

 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
T0*
_output_shapes
: 
Å
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
   
   *
_output_shapes
:*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
·
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *7æ*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
·
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *7?*
_output_shapes
: *
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0

Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
seed2 *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
T0*
_output_shapes

:

*
dtype0*

seed 

>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
¬
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*
_output_shapes

:

*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0

:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes

:

*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
ß
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*0
shared_name!dnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: *
dtype0*
shape
:

*
	container 

@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
Ų
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
Ē
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:

*
dtype0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
®
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
valueB
*    *
_output_shapes
:
*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
Õ
dnn/hiddenlayer_1/bias/part_0VarHandleOp*.
shared_namednn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: *
dtype0*
shape:
*
	container 

>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
Ē
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0
½
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
:
*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0

'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:

*
dtype0
v
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
T0*
_output_shapes

:


¬
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:’’’’’’’’’


%dnn/hiddenlayer_1/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
:
*
dtype0
n
dnn/hiddenlayer_1/biasIdentity%dnn/hiddenlayer_1/bias/ReadVariableOp*
T0*
_output_shapes
:


dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’

k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’

i
dnn/zero_fraction_1/SizeSizednn/hiddenlayer_1/Relu*
out_type0	*
T0*
_output_shapes
: 
e
dnn/zero_fraction_1/LessEqual/yConst*
valueB	 R’’’’*
_output_shapes
: *
dtype0	

dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction_1/cond/SwitchSwitchdnn/zero_fraction_1/LessEqualdnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_1/cond/switch_tIdentity!dnn/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_1/cond/switch_fIdentitydnn/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 

,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
Õ
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*
T0*'
_output_shapes
:’’’’’’’’’

ź
6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*:
_output_shapes(
&:’’’’’’’’’
:’’’’’’’’’
*)
_class
loc:@dnn/hiddenlayer_1/Relu
µ
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*

DstT0*'
_output_shapes
:’’’’’’’’’
*
Truncate( *

SrcT0

”
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
valueB"       *
_output_shapes
:*
dtype0
Ō
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0

dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*

DstT0	*
_output_shapes
: *
Truncate( *

SrcT0

.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
valueB
 *    *
_output_shapes
: *
dtype0
Ł
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:’’’’’’’’’

ģ
8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*:
_output_shapes(
&:’’’’’’’’’
:’’’’’’’’’
*)
_class
loc:@dnn/hiddenlayer_1/Relu
¹
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*

DstT0	*'
_output_shapes
:’’’’’’’’’
*
Truncate( *

SrcT0

£
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
valueB"       *
_output_shapes
:*
dtype0
Ś
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
	keep_dims( *
T0	*
_output_shapes
: *

Tidx0
Ŗ
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
T0	*
N*
_output_shapes
: : 

*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
T0	*
_output_shapes
: 

+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0	

-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0	
¶
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
_output_shapes
: *
dtype0
±
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
_output_shapes
: *
dtype0

 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
T0*
_output_shapes
: 
·
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"
      *
_output_shapes
:*
dtype0*+
_class!
loc:@dnn/logits/kernel/part_0
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *=æ*
_output_shapes
: *
dtype0*+
_class!
loc:@dnn/logits/kernel/part_0
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *=?*
_output_shapes
: *
dtype0*+
_class!
loc:@dnn/logits/kernel/part_0

Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
seed2 *+
_class!
loc:@dnn/logits/kernel/part_0*
T0*
_output_shapes

:
*
dtype0*

seed 
ž
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes
: *+
_class!
loc:@dnn/logits/kernel/part_0

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*
_output_shapes

:
*+
_class!
loc:@dnn/logits/kernel/part_0

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*
_output_shapes

:
*+
_class!
loc:@dnn/logits/kernel/part_0
Ź
dnn/logits/kernel/part_0VarHandleOp*)
shared_namednn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
dtype0*
shape
:
*
	container 

9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
¼
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
²
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:
*
dtype0*+
_class!
loc:@dnn/logits/kernel/part_0
 
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
dtype0*)
_class
loc:@dnn/logits/bias/part_0
Ą
dnn/logits/bias/part_0VarHandleOp*'
shared_namednn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
: *
dtype0*
shape:*
	container 
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
«
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*)
_class
loc:@dnn/logits/bias/part_0*
dtype0
Ø
*dnn/logits/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
_output_shapes
:*
dtype0*)
_class
loc:@dnn/logits/bias/part_0
y
 dnn/logits/kernel/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:
*
dtype0
h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
T0*
_output_shapes

:


dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:’’’’’’’’’
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
_output_shapes
:*
dtype0
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
T0*
_output_shapes
:

dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*
data_formatNHWC*'
_output_shapes
:’’’’’’’’’
e
dnn/zero_fraction_2/SizeSizednn/logits/BiasAdd*
out_type0	*
T0*
_output_shapes
: 
e
dnn/zero_fraction_2/LessEqual/yConst*
valueB	 R’’’’*
_output_shapes
: *
dtype0	

dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction_2/cond/SwitchSwitchdnn/zero_fraction_2/LessEqualdnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_2/cond/switch_tIdentity!dnn/zero_fraction_2/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_2/cond/switch_fIdentitydnn/zero_fraction_2/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_2/cond/pred_idIdentitydnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: 

,dnn/zero_fraction_2/cond/count_nonzero/zerosConst"^dnn/zero_fraction_2/cond/switch_t*
valueB
 *    *
_output_shapes
: *
dtype0
Õ
/dnn/zero_fraction_2/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_2/cond/count_nonzero/zeros*
T0*'
_output_shapes
:’’’’’’’’’
ā
6dnn/zero_fraction_2/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*
T0*:
_output_shapes(
&:’’’’’’’’’:’’’’’’’’’*%
_class
loc:@dnn/logits/BiasAdd
µ
+dnn/zero_fraction_2/cond/count_nonzero/CastCast/dnn/zero_fraction_2/cond/count_nonzero/NotEqual*

DstT0*'
_output_shapes
:’’’’’’’’’*
Truncate( *

SrcT0

”
,dnn/zero_fraction_2/cond/count_nonzero/ConstConst"^dnn/zero_fraction_2/cond/switch_t*
valueB"       *
_output_shapes
:*
dtype0
Ō
4dnn/zero_fraction_2/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_2/cond/count_nonzero/Cast,dnn/zero_fraction_2/cond/count_nonzero/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0

dnn/zero_fraction_2/cond/CastCast4dnn/zero_fraction_2/cond/count_nonzero/nonzero_count*

DstT0	*
_output_shapes
: *
Truncate( *

SrcT0

.dnn/zero_fraction_2/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_2/cond/switch_f*
valueB
 *    *
_output_shapes
: *
dtype0
Ł
1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_2/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:’’’’’’’’’
ä
8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_2/cond/pred_id*
T0*:
_output_shapes(
&:’’’’’’’’’:’’’’’’’’’*%
_class
loc:@dnn/logits/BiasAdd
¹
-dnn/zero_fraction_2/cond/count_nonzero_1/CastCast1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual*

DstT0	*'
_output_shapes
:’’’’’’’’’*
Truncate( *

SrcT0

£
.dnn/zero_fraction_2/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_2/cond/switch_f*
valueB"       *
_output_shapes
:*
dtype0
Ś
6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_2/cond/count_nonzero_1/Cast.dnn/zero_fraction_2/cond/count_nonzero_1/Const*
	keep_dims( *
T0	*
_output_shapes
: *

Tidx0
Ŗ
dnn/zero_fraction_2/cond/MergeMerge6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_2/cond/Cast*
T0	*
N*
_output_shapes
: : 

*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Sizednn/zero_fraction_2/cond/Merge*
T0	*
_output_shapes
: 

+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0	

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0	
¶
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
T0*
_output_shapes
: 

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
_output_shapes
: *
dtype0
£
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
_output_shapes
: *
dtype0

dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
T0*
_output_shapes
: 
s
!dnn/head/predictions/logits/ShapeShapednn/logits/BiasAdd*
out_type0*
T0*
_output_shapes
:
w
5dnn/head/predictions/logits/assert_rank_at_least/rankConst*
value	B :*
_output_shapes
: *
dtype0
g
_dnn/head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
X
Pdnn/head/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
n
dnn/head/predictions/logisticSigmoiddnn/logits/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
r
dnn/head/predictions/zeros_like	ZerosLikednn/logits/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
u
*dnn/head/predictions/two_class_logits/axisConst*
valueB :
’’’’’’’’’*
_output_shapes
: *
dtype0
Ł
%dnn/head/predictions/two_class_logitsConcatV2dnn/head/predictions/zeros_likednn/logits/BiasAdd*dnn/head/predictions/two_class_logits/axis*
T0*
N*'
_output_shapes
:’’’’’’’’’*

Tidx0

"dnn/head/predictions/probabilitiesSoftmax%dnn/head/predictions/two_class_logits*
T0*'
_output_shapes
:’’’’’’’’’
s
(dnn/head/predictions/class_ids/dimensionConst*
valueB :
’’’’’’’’’*
_output_shapes
: *
dtype0
Ę
dnn/head/predictions/class_idsArgMax%dnn/head/predictions/two_class_logits(dnn/head/predictions/class_ids/dimension*
output_type0	*
T0*#
_output_shapes
:’’’’’’’’’*

Tidx0
n
#dnn/head/predictions/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
_output_shapes
: *
dtype0
°
dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/class_ids#dnn/head/predictions/ExpandDims/dim*
T0	*

Tdim0*'
_output_shapes
:’’’’’’’’’
Ż
 dnn/head/predictions/str_classesAsStringdnn/head/predictions/ExpandDims*
	precision’’’’’’’’’*
shortest( *

scientific( *
T0	*'
_output_shapes
:’’’’’’’’’*
width’’’’’’’’’*

fill 
p
dnn/head/ShapeShape"dnn/head/predictions/probabilities*
out_type0*
T0*
_output_shapes
:
f
dnn/head/strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
h
dnn/head/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
h
dnn/head/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
¦
dnn/head/strided_sliceStridedSlicednn/head/Shapednn/head/strided_slice/stackdnn/head/strided_slice/stack_1dnn/head/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
V
dnn/head/range/startConst*
value	B : *
_output_shapes
: *
dtype0
V
dnn/head/range/limitConst*
value	B :*
_output_shapes
: *
dtype0
V
dnn/head/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0

dnn/head/rangeRangednn/head/range/startdnn/head/range/limitdnn/head/range/delta*
_output_shapes
:*

Tidx0
°
dnn/head/AsStringAsStringdnn/head/range*
	precision’’’’’’’’’*
shortest( *

scientific( *
T0*
_output_shapes
:*
width’’’’’’’’’*

fill 
Y
dnn/head/ExpandDims/dimConst*
value	B : *
_output_shapes
: *
dtype0

dnn/head/ExpandDims
ExpandDimsdnn/head/AsStringdnn/head/ExpandDims/dim*
T0*

Tdim0*
_output_shapes

:
[
dnn/head/Tile/multiples/1Const*
value	B :*
_output_shapes
: *
dtype0

dnn/head/Tile/multiplesPackdnn/head/strided_slicednn/head/Tile/multiples/1*

axis *
T0*
N*
_output_shapes
:

dnn/head/TileTilednn/head/ExpandDimsdnn/head/Tile/multiples*
T0*'
_output_shapes
:’’’’’’’’’*

Tmultiples0

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
r
save/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:
*
dtype0
X
save/IdentityIdentitysave/Read/ReadVariableOp*
T0*
_output_shapes
:

^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:

z
save/Read_1/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
*
dtype0
`
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes

:

d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes

:

t
save/Read_2/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
:
*
dtype0
\
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes
:

`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes
:

z
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:

*
dtype0
`
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes

:


d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes

:


m
save/Read_4/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
_output_shapes
:*
dtype0
\
save/Identity_8Identitysave/Read_4/ReadVariableOp*
T0*
_output_shapes
:
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes
:
s
save/Read_5/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:
*
dtype0
a
save/Identity_10Identitysave/Read_5/ReadVariableOp*
T0*
_output_shapes

:

f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:


save/StringJoin/inputs_1Const*<
value3B1 B+_temp_2d8b5bce543c4cfea6cb4f82ae7340f3/part*
_output_shapes
: *
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
{
save/SaveV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
_output_shapes
:*
dtype0
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
_output_shapes
: *
dtype0

save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/Read_6/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
_output_shapes
:
*
dtype0
l
save/Identity_12Identitysave/Read_6/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes
:


save/Read_7/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
_output_shapes

:
*
dtype0
p
save/Identity_14Identitysave/Read_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:

f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes

:


save/Read_8/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
_output_shapes
:
*
dtype0
l
save/Identity_16Identitysave/Read_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:

b
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes
:


save/Read_9/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
_output_shapes

:

*
dtype0
p
save/Identity_18Identitysave/Read_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:


f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:


}
save/Read_10/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
m
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
T0*
_output_shapes
:

save/Read_11/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
_output_shapes

:
*
dtype0
q
save/Identity_22Identitysave/Read_11/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:

f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:

ū
save/SaveV2_1/tensor_namesConst"/device:CPU:0*
valueBBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
_output_shapes
:*
dtype0
¼
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*[
valueRBPB10 0,10B7 10 0,7:0,10B10 0,10B10 10 0,10:0,10B1 0,1B10 1 0,10:0,1*
_output_shapes
:*
dtype0
ü
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_13save/Identity_15save/Identity_17save/Identity_19save/Identity_21save/Identity_23"/device:CPU:0*
dtypes

2
Ø
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*
_output_shapes
: *)
_class
loc:@save/ShardedFilename_1
ą
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*

axis *
T0*
N*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
Ø
save/Identity_24Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
~
save/RestoreV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
_output_shapes
:*
dtype0
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*
_output_shapes
:

save/AssignAssignglobal_stepsave/RestoreV2*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@global_step
(
save/restore_shardNoOp^save/Assign
ž
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*
valueBBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
_output_shapes
:*
dtype0
æ
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*[
valueRBPB10 0,10B7 10 0,7:0,10B10 0,10B10 10 0,10:0,10B1 0,1B10 1 0,10:0,1*
_output_shapes
:*
dtype0
Ö
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes

2*D
_output_shapes2
0:
:
:
:

::

b
save/Identity_25Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes
:

v
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_25"/device:CPU:0*
dtype0
h
save/Identity_26Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes

:

z
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_26"/device:CPU:0*
dtype0
d
save/Identity_27Identitysave/RestoreV2_1:2"/device:CPU:0*
T0*
_output_shapes
:

x
save/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_27"/device:CPU:0*
dtype0
h
save/Identity_28Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes

:


z
save/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_28"/device:CPU:0*
dtype0
d
save/Identity_29Identitysave/RestoreV2_1:4"/device:CPU:0*
T0*
_output_shapes
:
q
save/AssignVariableOp_4AssignVariableOpdnn/logits/bias/part_0save/Identity_29"/device:CPU:0*
dtype0
h
save/Identity_30Identitysave/RestoreV2_1:5"/device:CPU:0*
T0*
_output_shapes

:

s
save/AssignVariableOp_5AssignVariableOpdnn/logits/kernel/part_0save/Identity_30"/device:CPU:0*
dtype0
Å
save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"?
save/Const:0save/Identity_24:0save/restore_all (5 @F8"%
saved_model_main_op


group_deps"ß 
cond_contextĪ Ė 
¬
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *Ą
dnn/hiddenlayer_0/Relu:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0R
dnn/hiddenlayer_0/Relu:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1

"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*Æ
dnn/hiddenlayer_0/Relu:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0T
dnn/hiddenlayer_0/Relu:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ź
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *Ų
dnn/hiddenlayer_1/Relu:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0T
dnn/hiddenlayer_1/Relu:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0
·
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*Å
dnn/hiddenlayer_1/Relu:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0V
dnn/hiddenlayer_1/Relu:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0
Ā
"dnn/zero_fraction_2/cond/cond_text"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_t:0 *Š
dnn/logits/BiasAdd:0
dnn/zero_fraction_2/cond/Cast:0
-dnn/zero_fraction_2/cond/count_nonzero/Cast:0
.dnn/zero_fraction_2/cond/count_nonzero/Const:0
8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_2/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_2/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_2/cond/count_nonzero/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_t:0P
dnn/logits/BiasAdd:08dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0
Æ
$dnn/zero_fraction_2/cond/cond_text_1"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_f:0*½
dnn/logits/BiasAdd:0
/dnn/zero_fraction_2/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_2/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_2/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_f:0R
dnn/logits/BiasAdd:0:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0"’

	variablesń
ī

Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
ģ
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel
  "
(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias
 "
(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
ģ
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel

  "

(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias
 "
(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
É
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel
  "
(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
³
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"­

trainable_variables


ģ
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel
  "
(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias
 "
(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
ģ
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel

  "

(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias
 "
(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
É
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel
  "
(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
³
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"
	summariesō
ń
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0*µ
predict©
5
examples)
input_example_tensor:0’’’’’’’’’E
	class_ids8
!dnn/head/predictions/ExpandDims:0	’’’’’’’’’L
probabilities;
$dnn/head/predictions/probabilities:0’’’’’’’’’D
classes9
"dnn/head/predictions/str_classes:0’’’’’’’’’5
logits+
dnn/logits/BiasAdd:0’’’’’’’’’B
logistic6
dnn/head/predictions/logistic:0’’’’’’’’’tensorflow/serving/predict*£

regression
3
inputs)
input_example_tensor:0’’’’’’’’’A
outputs6
dnn/head/predictions/logistic:0’’’’’’’’’tensorflow/serving/regress*ą
serving_defaultĢ
3
inputs)
input_example_tensor:0’’’’’’’’’1
classes&
dnn/head/Tile:0’’’’’’’’’E
scores;
$dnn/head/predictions/probabilities:0’’’’’’’’’tensorflow/serving/classify*ß
classificationĢ
3
inputs)
input_example_tensor:0’’’’’’’’’1
classes&
dnn/head/Tile:0’’’’’’’’’E
scores;
$dnn/head/predictions/probabilities:0’’’’’’’’’tensorflow/serving/classify