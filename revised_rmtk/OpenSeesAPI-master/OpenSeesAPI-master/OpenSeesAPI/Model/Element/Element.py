"""
This class is used to create the following OpenSees TCL Commands:

This command is used to construct an element and add it to the Domain.
element eleType? arg1? ...

The type of element created and the additional arguments required depends on the eleType? provided in the command.

NOTE:
The valid queries to any element when creating an ElementRecorder are documented in the NOTES section for each element.

The following contain information about eleType? and the args required for each of the available element types:

Zero-Length Elements
zeroLength Element
zeroLengthND Element
zeroLengthSection Element
CoupledZeroLength Element
zeroLengthContact Element
zeroLengthContactNTS2D
zeroLengthInterface2D
zeroLengthImpact3D
Truss Elements
Truss Element
Corotational Truss Element
Beam-Column Elements
Elastic Beam Column Element
Elastic Beam Column Element with Stiffness Modifiers
Elastic Timoshenko Beam Column Element
Beam With Hinges Element
Displacement-Based Beam-Column Element
Force-Based Beam-Column Element
Flexure-Shear Interaction Displacement-Based Beam-Column Element
Joint Elements
BeamColumnJoint Element
ElasticTubularJoint Element
Joint2D Element
Link Elements
Two Node Link Element
Bearing Elements
Elastomeric Bearing (Plasticity) Element
Elastomeric Bearing (Bouc-Wen) Element
Flat Slider Bearing Element
Single Friction Pendulum Bearing Element
TFP Bearing
Triple Friction Pendulum Element
MultipleShearSpring Element
KikuchiBearing Element
YamamotoBiaxialHDR Element
ElastomericX
LeadRubberX
HDR
RJ-Watson EQS Bearing Element
Quadrilateral Elements
Quad Element
Shell Element
ShellNL
Bbar Plane Strain Quadrilateral Element
Enhanced Strain Quadrilateral Element
SSPquad Element
Triangular Elements
Tri31 Element
Brick Elements
Standard Brick Element
Bbar Brick Element
Twenty Node Brick Element
Twenty Seven Node Brick Element
SSPbrick Element
u-p Elements
UC San Diego u-p element (saturated soil)
Four Node Quad u-p Element
Brick u-p Element
bbarQuad u-p Element
bbarBrick u-p Element
Nine Four Node Quad u-p Element
Twenty Eight Node Brick u-p Element
Twenty Node Brick u-p Element
Brick Large Displacement u-p Element
SSPquadUP Element
SSPbrickUP Element
Misc.
ShallowFoundationGen
SurfaceLoad Element
Contact Elements
SimpleContact2D Element
SimpleContact3D Element
BeamContact2D Element
BeamContact3D Element
BeamEndContact3D Element
zeroLengthImpact3D
"""

__author__ = 'marafi'


from OpenSeesAPI.OpenSees import OpenSees

class Truss(OpenSees):
    """
    One way is to specify an area and a UniaxialMaterial identifier:
    element truss $eleTag $iNode $jNode $A $matTag <-rho $rho> <-cMass $cFlag> <-doRayleigh $rFlag>
    the other is to specify a Section identifier:
    element trussSection $eleTag $iNode $jNode $secTag <-rho $rho> <-cMass $cFlag> <-doRayleigh $rFlag>

    $eleTag	unique element object tag
    $iNode $jNode	end nodes
    $A	cross-sectional area of element
    $matTag	tag associated with previously-defined UniaxialMaterial
    $secTag	tag associated with previously-defined Section
    $rho	mass per unit length, optional, default = 0.0
    $cFlag	consistent mass flag, optional, default = 0
    cFlag = 0 lumped mass matrix (default)
    cFlag = 1 consistent mass matrix
    $rFlag	Rayleigh damping flag, optional, default = 0
    rFlag = 0 NO RAYLEIGH DAMPING (default)
    rFlag = 1 include Rayleigh damping
    """
    def __init__(self, id, NodeI, NodeJ, Area, Material, **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._Area = Area
        self._Material = Material
        self._CommandLine =  'element truss %d %s %s %f %s'%(self.id, self._NodeI.id, self._NodeJ.id, self._Area, self._Material.id)
        self.__dict__.update(kwargs)

class ZeroLength(OpenSees):
    """
    element zeroLength $eleTag $iNode $jNode -mat $matTag1 $matTag2 ... -dir $dir1 $dir2 ...<-doRayleigh $rFlag> <-orient $x1 $x2 $x3 $yp1 $yp2 $yp3>

    $eleTag	unique element object tag
    $iNode $jNode	end nodes
    $matTag1 $matTag2 ...	tags associated with previously-defined UniaxialMaterials
    $dir1 $dir2 ...	material directions:
    1,2,3 - translation along local x,y,z axes, respectively;
    4,5,6 - rotation about local x,y,z axes, respectively
    $x1 $x2 $x3	vector components in global coordinates defining local x-axis (optional)
    $yp1 $yp2 $yp3	vector components in global coordinates defining vector yp which lies in the local x-y plane for the element. (optional)
    $rFlag	optional, default = 0
    rFlag = 0 NO RAYLEIGH DAMPING (default)
    rFlag = 1 include rayleigh damping
    """
    def __init__(self, id, NodeI, NodeJ, MaterialList, DOFList, OrientDirection=None, **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._MaterialList = MaterialList
        self._DOFList = DOFList
        self._OrientDirection = OrientDirection
        self.__dict__.update(kwargs)
        OD = ''
        if OrientDirection != None:
            OD = '-orient %f %f %f %f %f %f'%tuple(OrientDirection)
        self._CommandLine = 'element zeroLength %d %d %d -mat %s -dir %s %s'%(self._id, self._NodeI.id, self._NodeJ.id, ''.join([' %d'%s.id for s in MaterialList]), ''.join([' %d'%s for s in DOFList]), OD)

class ZeroLengthSection(OpenSees):
    """
    This command is used to construct a zero length element object, which is defined by two nodes at the same location. The nodes are connected by a single section object to represent the force-deformation relationship for the element.

    element zeroLengthSection $eleTag $iNode $jNode $secTag <-orient $x1 $x2 $x3 $yp1 $yp2 $yp3> <-doRayleigh $rFlag>

    $eleTag	unique element object tag
    $iNode $jNode	end nodes
    $secTag	tag associated with previously-defined Section object
    $x1 $x2 $x3	vector components in global coordinates defining local x-axis (optional)
    $yp1 $yp2 $yp3	vector components in global coordinates defining vector yp which lies in the local x-y plane for the element. (optional)
    $rFlag	optional, default = 1
    rFlag = 0 no Rayleigh damping
    rFlag = 1 include Rayleigh damping (default)
    """
    def __init__(self, id, NodeI, NodeJ, Section, OrientDirection=None, Optional='',**kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._Section = Section
        self._OrientDirection = OrientDirection
        self._Optional = Optional
        self.__dict__.update(kwargs)
        OD = ''
        if OrientDirection != None:
            OD = '-orient %f %f %f %f %f %f'%tuple(OrientDirection)
        self._CommandLine = 'element zeroLengthSection %d %d %d %d %s %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._Section.id, OD, self._Optional)

class ElasticBeamColumn(OpenSees):
    """
    For a two-dimensional problem:
    element elasticBeamColumn $eleTag $iNode $jNode $A $E $Iz $transfTag <-mass $massDens> <-cMass>
    For a three-dimensional problem:
    element elasticBeamColumn $eleTag $iNode $jNode $A $E $G $J $Iy $Iz $transfTag <-mass $massDens> <-cMass>

    $eleTag	unique element object tag
    $iNode $jNode	end nodes
    $A	cross-sectional area of element
    $E	Young's Modulus
    $G	Shear Modulus
    $J	torsional moment of inertia of cross section
    $Iz	second moment of area about the local z-axis
    $Iy	second moment of area about the local y-axis
    $transfTag	identifier for previously-defined coordinate-transformation (CrdTransf) object
    $massDens	element mass per unit length (optional, default = 0.0)
    -cMass	to form consistent mass matrix (optional, default = lumped mass matrix)
    """
    def __init__(self, id, NodeI, NodeJ, A, E, Iz, TransTag, G=None, J=None, Iy=None, Mass=None, Option='', **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._A = A
        self._E = E
        self._Iz = Iz
        self._TransTag = TransTag
        self._G = G
        self._J = J
        self._Iy = Iy
        self._Mass = Mass
        self._Option = Option
        self.__dict__.update(kwargs)

        if self._Mass != None:
            self._EndCommand = '-mass %f %s'%(self._Mass,self._Option)
        else:
            self._EndCommand = self._Option

        if G == None:
            self._CommandLine = 'element elasticBeamColumn %d %d %d %f %f %f %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._A, self._E, self._Iz, self._TransTag.id, self._EndCommand)
        else:
            self._CommandLine = 'element elasticBeamColumn %d %d %d %f %f %f %f %f %f %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._A, self._E, self._G, self._J, self._Iy, self._Iz, self._TransTag.id, self._EndCommand)


class ElasticTimoshenkoBeam(OpenSees):
    """
    This command is used to construct an ElasticTimoshenkoBeam element object. A Timoshenko beam is a frame member that accounts for shear deformations. The arguments for the construction of an elastic Timoshenko beam element depend on the dimension of the problem, ndm:
    For a two-dimensional problem:
    element ElasticTimoshenkoBeam $eleTag $iNode $jNode $E $G $A $Iz $Avy $transfTag <-mass $massDens> <-cMass>
    For a three-dimensional problem:
    element ElasticTimoshenkoBeam $eleTag $iNode $jNode $E $G $A $Jx $Iy $Iz $Avy $Avz $transfTag <-mass $massDens> <-cMass>

    $eleTag	unique element object tag
    $iNode $jNode	end nodes
    $E	Young's Modulus
    $G	Shear Modulus
    $A	cross-sectional area of element
    $Jx	torsional moment of inertia of cross section
    $Iy	second moment of area about the local y-axis
    $Iz	second moment of area about the local z-axis
    $Avy	Shear area for the local y-axis
    $Avz	Shear area for the local z-axis
    $transfTag	identifier for previously-defined coordinate-transformation (CrdTransf) object
    $massDens	element mass per unit length (optional, default = 0.0)
    -cMass	to form consistent mass matrix (optional, default = lumped mass matrix)
    """
    def __init__(self, id, NodeI, NodeJ, E, G, A, Iz, Avy, TransTag, Jx=None, Iy=None, Avz=None, MassDensity=None, Option='', **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._E = E
        self._G = G
        self._A = A
        self._Jx = Jx
        self._Iy = Iy
        self._Iz = Iz
        self._Avy = Avy
        self._Avz = Avz
        self._TransTag = TransTag
        self._MassDensity = MassDensity
        self._Option = Option
        self.__dict__.update(kwargs)

        if self._MassDensity != None:
            self._EndCommand = '-mass %f %s'%(self._MassDensity,self._Option)
        else:
            self._EndCommand = self._Option

        if Jx == None:
            self._CommandLine = 'element ElasticTimoshenkoBeam %d %d %d %f %f %f %f %f %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._E, self._G, self._A, self._Iz, self._Avy, self._TransTag.id, self._EndCommand)
        else:
            self._CommandLine = 'element ElasticTimoshenkoBeam %d %d %d %f %f %f %f %f %f %f %f %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._E, self._G, self._A, self._Jx, self._Iy, self._Iz, self._Avy, self._Avz, self._TransTag.id, self._EndCommand)


class DispBeamColumn(OpenSees):
    """
    element dispBeamColumn $eleTag $iNode $jNode $numIntgrPts $secTag $transfTag <-mass $massDens> <-cMass> <-integration $intType>
    To change the sections along the element length, the following form of command may be used:
    element dispBeamColumn $eleTag $iNode $jNode $numIntgrPts -sections $secTag1 $secTag2 ... $transfTag <-mass $massDens> <-cMass> <-integration $intType>

    $eleTag	unique element object tag
    $iNode $jNode	end nodes
    $numIntgrPts	number of integration points along the element.
    $secTag	identifier for previously-defined section object
    $secTag1 $secTag2 ...	$numIntgrPts identifiers of previously-defined section object
    $transfTag	identifier for previously-defined coordinate-transformation (CrdTransf) object
    $massDens	element mass density (per unit length), from which a lumped-mass matrix is formed (optional, default = 0.0)
    -cMass	to form consistent mass matrix (optional, default = lumped mass matrix)
    $intType	numerical integration type, options are Lobotto, Legendre, Radau, NewtonCotes, Trapezoidal (optional, default = Legendre)
    """
    def __init__(self, id, NodeI, NodeJ, numIntgrPts, Section, GeomTrans, Mass=None, Optional='', **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._numIntgrPts = numIntgrPts
        self._Section = Section
        self._GeoTrans = GeomTrans
        self._Mass = Mass
        self._Optional = Optional
        self.__dict__.update(kwargs)

        if self._Mass != None:
            self._EndCommand = '-mass %f %s'%(self._Mass,self._Optional)
        else:
            self._EndCommand = self._Optional
        if type(self._Section) != list:
            self._CommandLine = 'element dispBeamColumn %d %d %d %d %d %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._numIntgrPts, self._Section.id, self._GeoTrans.id, self._EndCommand)
        else:
            self._CommandLine = 'element dispBeamColumn %d %d %d %d -sections %s %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._numIntgrPts, ''.join([' %d'%x.id for x in self._Section]), self._GeoTrans.id, self._EndCommand)

class ForceBeamColumn(OpenSees):
    """
    element forceBeamColumn $eleTag $iNode $jNode $transfTag "IntegrationType arg1 arg2 ..." <-mass $massDens> <-iter $maxIters $tol>
    $eleTag	unique element object tag
    $iNode $jNode	end nodes
    $transfTag	identifier for previously-defined coordinate-transformation (CrdTransf) object
    IntegrationType arg1 arg2 ...	specifies locations and weights of integration points and their associated section force-deformation models (see File:IntegrationTypes.pdf)
    $massDens	element mass density (per unit length), from which a lumped-mass matrix is formed (optional, default=0.0)
    $maxIters	maximum number of iterations to undertake to satisfy element compatibility (optional, default=10)
    $tol	tolerance for satisfaction of element compatibility (optional, default=10-12)

    Original command that assumes Gauss-Lobatto integration with a copy of the same section force-deformation model at each integration point:
    element forceBeamColumn $eleTag $iNode $jNode $numIntgrPts $secTag $transfTag <-mass $massDens> <-iter $maxIters $tol> <-integration $intType>
    $eleTag	unique element object tag
    $numIntgrPts	number of Gauss-Lobatto integration points along the element.
    $secTag	identifier for previously-defined section object

    Alternative command (kept for backward compatability):
    element nonlinearBeamColumn $eleTag $iNode $jNode $numIntgrPts $secTag $transfTag <-mass $massDens> <-iter $maxIters $tol> <-integration $intType>
    $eleTag	unique element object tag
    $intType	numerical integration type, options are Lobatto, Legendre, Radau, NewtonCotes, Trapezoidal (optional, default= Lobatto)
    """
    def __init__(self, id, NodeI, NodeJ, GeomTrans, IntegrationType, Mass=None, Optional='', **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._GeomTrans = GeomTrans
        self._IntegrationType = IntegrationType # Assume to be a string of args
        self._Mass = Mass
        self._Optional = Optional
        self.__dict__.update(kwargs)

        if self._Mass != None:
            self._EndCommand = '-mass %f %s'%(self._Mass,self._Optional)
        else:
            self._EndCommand = ' %s'%(self._Optional)

        self._CommandLine = 'element forceBeamColumn %d %d %d %d %s %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._GeomTrans.id, self._IntegrationType, self._EndCommand)

class ForceBeamColumnOriginal(OpenSees):
    """
    Original command that assumes Gauss-Lobatto integration with a copy of the same section force-deformation model at each integration point:
    element forceBeamColumn $eleTag $iNode $jNode $numIntgrPts $secTag $transfTag <-mass $massDens> <-iter $maxIters $tol> <-integration $intType>
    $eleTag	unique element object tag
    $numIntgrPts	number of Gauss-Lobatto integration points along the element.
    $secTag	identifier for previously-defined section object
    """
    def __init__(self, id, NodeI, NodeJ, NoOfIntPoints, Section, GeomTrans, Mass=None, Optional='', **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._NoOfIntPoints = NoOfIntPoints
        self._Section = Section
        self._GeomTrans = GeomTrans
        self._Mass = Mass
        self._Optional = Optional
        self.__dict__.update(kwargs)

        if self._Mass != None:
            self._EndCommand = '-mass %f %s'%(self._Mass,self._Optional)
        else:
            self._EndCommand = ' %s'%(self._Optional)

        if type(self._Section) == str:
            self._CommandLine = 'element forceBeamColumn %d %d %d %d %s %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._NoOfIntPoints, self._Section, self._GeomTrans.id, self._EndCommand)
        elif type(self._Section) == list:
            self._CommandLine = 'element forceBeamColumn %d %d %d %d -sections %s %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._NoOfIntPoints, ''.join([' %d'%x.id for x in self._Section]), self._GeomTrans.id, self._EndCommand)
        else:
            self._CommandLine = 'element forceBeamColumn %d %d %d %d %d %d %s'%(self._id, self._NodeI.id, self._NodeJ.id, self._NoOfIntPoints, self._Section.id, self._GeomTrans.id, self._EndCommand)

class ForceBeamColumnUserDefined(OpenSees):
    """
    element forceBeamColumn $eleTag $iNode $jNode $transfTag "IntegrationType arg1 arg2 ..." <-mass $massDens> <-iter $maxIters $tol>
    $eleTag	unique element object tag
    $iNode $jNode	end nodes
    $transfTag	identifier for previously-defined coordinate-transformation (CrdTransf) object
    IntegrationType arg1 arg2 ...	specifies locations and weights of integration points and their associated section force-deformation models (see File:IntegrationTypes.pdf)
    $massDens	element mass density (per unit length), from which a lumped-mass matrix is formed (optional, default=0.0)
    $maxIters	maximum number of iterations to undertake to satisfy element compatibility (optional, default=10)
    $tol	tolerance for satisfaction of element compatibility (optional, default=10-12)
    Original command that assumes Gauss-Lobatto integration with a copy of the same section force-deformation model at each integration point:
    element forceBeamColumn $eleTag $iNode $jNode $numIntgrPts $secTag $transfTag <-mass $massDens> <-iter $maxIters $tol> <-integration $intType>
    $eleTag	unique element object tag
    $numIntgrPts	number of Gauss-Lobatto integration points along the element.
    $secTag	identifier for previously-defined section object
    Alternative command (kept for backward compatability):
    element nonlinearBeamColumn $eleTag $iNode $jNode $numIntgrPts $secTag $transfTag <-mass $massDens> <-iter $maxIters $tol> <-integration $intType>
    $eleTag	unique element object tag
    $intType	numerical integration type, options are Lobatto, Legendre, Radau, NewtonCotes, Trapezoidal (optional, default= Lobatto)
    """

    def __init__(self, id, NodeI, NodeJ, GeomTrans, SectionList, **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._GeomTrans = GeomTrans
        self._NoOfIntPoints = len(SectionList)
        self._SectionList = SectionList #list of section ids
        self.__dict__.update(kwargs)
        self._SectionString = ''
        for tag in SectionList:
            self._SectionString += '%d '%tag._id

        if self._NoOfIntPoints == 3:
            self._CommandLine = 'element forceBeamColumn %d %d %d %d UserDefined %d %s 0 0.5 1 0.166665 0.66667 0.166665'%(self._id, self._NodeI.id, self._NodeJ.id, self._GeomTrans.id,self._NoOfIntPoints,self._SectionString)
        elif self._NoOfIntPoints == 4:
            self._CommandLine = 'element forceBeamColumn %d %d %d %d UserDefined %d %s 0 0.276395 0.723605 1 0.083334 0.416665 0.416665 0.083334'%(self._id, self._NodeI.id, self._NodeJ.id, self._GeomTrans.id,self._NoOfIntPoints,self._SectionString)
        elif self._NoOfIntPoints == 5:
            self._CommandLine = 'element forceBeamColumn %d %d %d %d UserDefined %d %s 0 0.17267 0.5 0.82733 1 0.05 0.272222 0.355556 0.272222 0.05 '%(self._id, self._NodeI.id, self._NodeJ.id, self._GeomTrans.id,self._NoOfIntPoints,self._SectionString)
        elif self._NoOfIntPoints == 6:
            self._CommandLine = 'element forceBeamColumn %d %d %d %d UserDefined %d %s 0 0.117473 0.357384 0.642616 0.882528 1 0.033333 0.189235 0.27743 0.27743 0.189235 0.033333'%(self._id, self._NodeI.id, self._NodeJ.id, self._GeomTrans.id, self._NoOfIntPoints, self._SectionString)
        else:
            print('Number of Sections not supported')
            self._CommandLine = ''

class Joint2D(OpenSees):
    """
    element Joint2D $eleTag $Nd1 $Nd2 $Nd3 $Nd4 $NdC <$Mat1 $Mat2 $Mat3 $Mat4> $MatC $LrgDspTag

    $eleTag	unique element object tag
    $Nd1 $Nd2 $Nd3 $Nd4	integer tags indicating four external nodes where the joint element is connected to the adjoining beam-column element
    $NdC	integer tags indicating the central node of beam-column joint (the tag is used to generate the internal node, thus, the node should not exist in the domain or be used by any other node)
    $Mat1	uniaxial material tag for interface rotational spring at node 1. Use a zero tag to indicate the case that a beam-column element is rigidly framed to the joint. (optional)
    $Mat2	uniaxial material tag for interface rotational spring at node 2. Use a zero tag to indicate the case that a beam-column element is rigidly framed to the joint. (optional)
    $Mat3	uniaxial material tag for interface rotational spring at node 3. Use a zero tag to indicate the case that a beam-column element is rigidly framed to the joint. (optional)
    $Mat4	uniaxial material tag for interface rotational spring at node 4. Use a zero tag to indicate the case that a beam-column element is rigidly framed to the joint. (optional)
    $MatC	uniaxial material tag for rotational spring of the central node that describes shear panel behavior
    $LrgDspTag	an integer indicating the flag for considering large deformations:
    0 - for small deformations and constant geometry
    1 - for large deformations and time varying geometry
    """
    def __init__(self, id, NodeI, NodeJ, NodeK, NodeL, NodeCTag, MatC, LargeDispTag, MatIJKL=None, **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._NodeK = NodeK
        self._NodeL = NodeL
        self._NodeCTag = NodeCTag
        self._MatC = MatC
        self._LargeDispTag = LargeDispTag
        self._MatIJKL = MatIJKL
        self.__dict__.update(kwargs)

        if MatIJKL==None:
            self._CommandLine = 'element Joint2D %d %d %d %d %d %d %d %d'%(self._id, self._NodeI.id, self._NodeJ.id, self._NodeK.id, self._NodeL.id, self._NodeCTag, self._MatC.id, self._LargeDispTag)
        else:
            self._CommandLine = 'element Joint2D %d %d %d %d %d %d %d %d %d %d %d %d'%(self._id, self._NodeI.id, self._NodeJ.id, self._NodeK.id, self._NodeL.id, self._NodeCTag, self._MatIJKL[0].id, self._MatIJKL[1].id, self._MatIJKL[2].id, self._MatIJKL[3].id,self._MatC.id, self._LargeDispTag)

class ShellMITC4(OpenSees):
    """
    This command is used to construct a ShellMITC4 element object, which uses a bilinear isoparametric formulation in combination with a modified shear interpolation to improve thin-plate bending performance.

    element ShellMITC4 $eleTag $iNode $jNode $kNode $lNode $secTag


    $eleTag	unique element object tag
    $iNode $jNode $kNode $lNode	four nodes defining element boundaries, input in counter-clockwise order around the element.
    $secTag	tag associated with previously-defined SectionForceDeformation object.
    Currently must be either a PlateFiberSection, or ElasticMembranePlateSection
    """
    def __init__(self, id, NodeI, NodeJ, NodeK, NodeL, Section, **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._NodeK = NodeK
        self._NodeL = NodeL
        self._Section = Section
        self.__dict__.update(kwargs)

        self._CommandLine = 'element ShellMITC4 %d %d %d %d %d %d'%(self._id, self._NodeI.id, self._NodeJ.id, self._NodeK.id, self._NodeL.id, self._Section.id)

class ShellDKGQ(OpenSees):
    """
    This command is used to construct a ShellDKGQ element object, which is a quadrilateral shell element based on the theory of generalized conforming element.

    element ShellDKGQ $eleTag $iNode $jNode $kNode $lNode $secTag


    $eleTag	unique element object tag
    $iNode $jNode $kNode $lNode	four nodes defining element boundaries, input in clockwise or counter-clockwise order around the element.
    $secTag	tag associated with previously-defined SectionForceDeformation object.
    Currently can be a PlateFiberSection, a ElasticMembranePlateSection and a LayeredShell section
    """

    def __init__(self, id, NodeI, NodeJ, NodeK, NodeL, Section, **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._NodeK = NodeK
        self._NodeL = NodeL
        self._Section = Section
        self.__dict__.update(kwargs)

        self._CommandLine = 'element ShellDKGQ %d %d %d %d %d %d' % (
        self._id, self._NodeI.id, self._NodeJ.id, self._NodeK.id, self._NodeL.id, self._Section.id)

class ShellNLDKGQ(OpenSees):
    """
    This command is used to construct a ShellNLDKGQ element object accounting for the geometric nonlinearity of large deformation using the updated Lagrangian formula, which is developed based on the ShellDKGQ element.

    element ShellNLDKGQ $eleTag $iNode $jNode $kNode $lNode $secTag


    $eleTag	unique element object tag
    $iNode $jNode $kNode $lNode	four nodes defining element boundaries, input in clockwise or counter-clockwise order around the element.
    $secTag	tag associated with previously-defined SectionForceDeformation object.
    Currently can be a PlateFiberSection, a ElasticMembranePlateSection and a LayeredShell section.
    """

    def __init__(self, id, NodeI, NodeJ, NodeK, NodeL, Section, **kwargs):
        self._id = id
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self._NodeK = NodeK
        self._NodeL = NodeL
        self._Section = Section
        self.__dict__.update(kwargs)

        self._CommandLine = 'element ShellNLDKGQ %d %d %d %d %d %d' % (
            self._id, self._NodeI.id, self._NodeJ.id, self._NodeK.id, self._NodeL.id, self._Section.id)

class MVLEM(OpenSees):
    """
    Element MVLEM $eleTag $Dens $iNode $jNode $m $c -thick {Thicknesses} -width {Widths} -rho {Reinforcing_ratios} -matConcrete {Concrete_tags} -matSteel {Steel_tags} -matShear {Shear_tag}
    $eleTag	Unique element object tag
    $Dens	Wall density
    $iNode $jNode	End node tags
    $m	Number of element macro-fibers
    $c	Location of center of rotation from the iNode, c = 0.4 (recommended)
    {Thicknesses}	Array of m macro-fiber thicknesses
    {Widths}	Array of m macro-fiber widths
    {Reinforcing_ratios}	Array of m reinforcing ratios corresponding to macro-fibers; for each fiber: rhoi = As,i/Agross,i (1 < i < m)
    {Concrete _tags}	Array of m uniaxialMaterial tags for concrete
    {Steel_tags}	Array of m uniaxialMaterial tags for steel
    {Shear_tag}	Tag of uniaxialMaterial for shear material
    """
    def __init__(self, id, Density, NodeI, NodeJ, m, c, Thicknesses, Widths, ReinforcingRatios, ConcreteTags, SteelTags, ShearTag, **kwargs):
        self._id = id
        self.Density = Density
        self._NodeI = NodeI
        self._NodeJ = NodeJ
        self.m = m
        self.c = c
        self.Thicknesses = Thicknesses
        self.Widths = Widths
        self.ReinforcingRatios = ReinforcingRatios
        self.ConcreteTags = ConcreteTags,
        self.SteelTags = SteelTags
        self.ShearTag = ShearTag
        self.__dict__.update(kwargs)

        self._CommandLine = 'element MVLEM %d %f %d %d %d %f -thick %s -width %s -rho %s -matConcrete %s -matSteel %s -matShear %d'%(self._id, self.Density, self._NodeI.id, self._NodeJ.id, self.m, self.c, ''.join([' %f'%s for s in self.Thicknesses]), ''.join([' %f'%s for s in self.Widths]), ''.join([' %f'%s for s in self.ReinforcingRatios]), ''.join([' %d'%s.id for s in self.ConcreteTags]), ''.join([' %d'%s.id for s in self.SteelTags]), self.ShearTag.id)
