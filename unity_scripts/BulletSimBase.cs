using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using BulletSharp;
using BulletSharp.Math;
using DemoFramework;
using BulletUnity;

/**
 *  Base class for creating and displaying a simulation using Bullet.
 * 	The "Sim" methods here are implemented as an example of how to use the base class, but
 * 		should be overriden in derived classes.
 */
public class BulletSimBase : MonoBehaviour {

	//
	// Publicly exposed members (in the editor).
	//
	public float simStepSize = 1.0f / 60.0f;
	public float simTimeScale = 1.0f;
	public GameObject ropePrefab;
	public GameObject softBodyPrefab;

	//
	// Internal members
	//

	// collision world things
	protected CollisionConfiguration m_colConfig;
	protected CollisionDispatcher m_colDispatcher;
	protected BroadphaseInterface m_broadphase;
	protected SequentialImpulseConstraintSolver m_solver;
	protected DynamicsWorld m_world;

	// bookkeeping
	protected List<CollisionShape> m_collisionShapes;
	protected List<GameObject> m_createdObjs;
	protected bool m_quitting = false;

	//
	// Overridden Unity methods.
	//

	// Use this for initialization
	void Start() {
		// unity stepping params
		SetStepSize(simStepSize);
		SetTimeScale(simTimeScale);

		// set up members
		m_collisionShapes = new List<CollisionShape>();
		m_createdObjs = new List<GameObject>();

		// set up the dynamics world
		// collision configuration contains default setup for memory, collision setup
		m_colConfig = new DefaultCollisionConfiguration();
		m_colDispatcher = new CollisionDispatcher(m_colConfig);
		m_broadphase = new DbvtBroadphase();
//		new AxisSweep3_32Bit(new Vector3(-10000, -10000, -10000), new Vector3(10000, 10000, 10000), 1024);
		m_solver = new SequentialImpulseConstraintSolver();

		m_world = new DiscreteDynamicsWorld(m_colDispatcher, m_broadphase, m_solver, m_colConfig);
		m_world.Gravity = new BulletSharp.Math.Vector3(0, -9.8f, 0);

		// set up simulation 
		InitSim();

		// Debug Draw for collision boundaries
		DebugDrawUnity m_debugDrawer = new DebugDrawUnity();
		m_debugDrawer.DebugMode = DebugDrawModes.DrawWireframe;
		m_world.DebugDrawer = m_debugDrawer;
		
	}
	
	// Update is called once per frame
	void Update() {
		if (m_quitting) return;
		RenderUpdateSim();
	}

	// Called at a fixed interval
	void FixedUpdate() {
		if (m_quitting) return;
		if (m_world != null)
		{
			m_world.StepSimulation(Time.fixedDeltaTime);
		}
		UpdateSim();
	}

	public void OnDrawGizmos()
	{
		if (m_quitting) return;
		if (m_world != null)
		{
			m_world.DebugDrawWorld();
		}
	}


	void OnDestroy() {
//		ExitSimulation();
	}

	//
	// Helpers
	// 

	protected void SetStepSize(float dt) {
		Time.fixedDeltaTime = dt;
	}

	protected void SetTimeScale(float timeScale) {
		Time.timeScale = timeScale;
	}

	protected DynamicsWorld GetDynamicsWorld() {
		return m_world;
	}

	protected void AddCollisionShape(CollisionShape shape) {
		m_collisionShapes.Add(shape);
	}

	protected void RemoveCollisionShape(CollisionShape shape) {
		m_collisionShapes.Remove(shape);
		shape.Dispose();
	}

	// Creates a rigid body from the given shape and adds it to the Unity scene.
	protected RigidBody CreateRigidBody(float mass, Matrix startTransform, CollisionShape shape, Material renderMat, float friction = 0.5f, bool isKinematic = false, bool viz = false)
	{
		//rigidbody is dynamic if and only if mass is non zero, otherwise static
		bool isDynamic = (mass != 0.0f);

		BulletSharp.Math.Vector3 localInertia = BulletSharp.Math.Vector3.Zero;
		if (isDynamic) {
			shape.CalculateLocalInertia(mass, out localInertia);
		}

		//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
		DefaultMotionState myMotionState = new DefaultMotionState(startTransform);

		RigidBodyConstructionInfo rbInfo = new RigidBodyConstructionInfo(mass, myMotionState, shape, localInertia);
		rbInfo.Friction = friction;
		RigidBody body = new RigidBody(rbInfo);
		if (isKinematic) {
			body.CollisionFlags = body.CollisionFlags | BulletSharp.CollisionFlags.KinematicObject;
			body.ActivationState = ActivationState.DisableDeactivation;
		}
		rbInfo.Dispose();

		m_world.AddRigidBody(body);

		// create unity object from it
		if (viz) AddUnityObject(body, renderMat);

		return body;
	}

	// Creates a rigid body from the given shape and adds it to the Unity scene.
	protected RigidBody CreateRigidBody(float mass, BulletSharp.Math.Vector3 inertia, Matrix startTransform, CollisionShape shape, Material renderMat, float friction = 0.5f, bool isKinematic = false, bool viz = false)
    {
        //rigidbody is dynamic if and only if mass is non zero, otherwise static
        bool isDynamic = (mass != 0.0f);

		BulletSharp.Math.Vector3 localInertia = BulletSharp.Math.Vector3.Zero;
		if (isDynamic) {
			localInertia = inertia;
		}

        //using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects
        DefaultMotionState myMotionState = new DefaultMotionState(startTransform);

        RigidBodyConstructionInfo rbInfo = new RigidBodyConstructionInfo(mass, myMotionState, shape, localInertia);
        rbInfo.Friction = friction;
		//rbInfo.RollingFriction = friction;
        RigidBody body = new RigidBody(rbInfo);
        if (isKinematic)
        {
            body.CollisionFlags = body.CollisionFlags | BulletSharp.CollisionFlags.KinematicObject;
            body.ActivationState = ActivationState.DisableDeactivation;
        }
        rbInfo.Dispose();

        m_world.AddRigidBody(body);

        // create unity object from it
        if (viz) AddUnityObject(body, renderMat);

        return body;
    }

	protected RigidBody ResetRigidBody(RigidBody rb, Matrix startTransform, CollisionShape shape, float friction = 0.5f, bool isKinematic = false) {
		// basically detroys a rigid body and re-initializes it efficiently
		// doesn't recalculate moment of inertia or re-create the gfx object

		//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects\
		float mass = rb.InvMass;
		if (mass != 0.0) mass = 1 / mass;
		BulletSharp.Math.Vector3 localInertia = rb.LocalInertia;
		DestroyRigidBody(rb);
		DefaultMotionState myMotionState = new DefaultMotionState(startTransform);

		RigidBodyConstructionInfo rbInfo = new RigidBodyConstructionInfo(mass, myMotionState, shape, localInertia);
		rbInfo.Friction = friction;
		RigidBody body = new RigidBody(rbInfo);
		if (isKinematic) {
			body.CollisionFlags = body.CollisionFlags | BulletSharp.CollisionFlags.KinematicObject;
			body.ActivationState = ActivationState.DisableDeactivation;
		}
		rbInfo.Dispose();

		m_world.AddRigidBody(body);

		return body;
	}

	protected RigidBody ResetRigidBody(RigidBody rb, float newMass, BulletSharp.Math.Vector3 newInertia, Matrix startTransform, CollisionShape shape, float friction = 0.5f, bool isKinematic = false) {
		// basically detroys a rigid body and re-initializes it efficiently
		// doesn't recalculate moment of inertia or re-create the gfx object

		//using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects\
		float mass = newMass;
		BulletSharp.Math.Vector3 localInertia = newInertia;
		DestroyRigidBody(rb);
		DefaultMotionState myMotionState = new DefaultMotionState(startTransform);

		RigidBodyConstructionInfo rbInfo = new RigidBodyConstructionInfo(mass, myMotionState, shape, localInertia);
		rbInfo.Friction = friction;
		RigidBody body = new RigidBody(rbInfo);
		if (isKinematic) {
			body.CollisionFlags = body.CollisionFlags | BulletSharp.CollisionFlags.KinematicObject;
			body.ActivationState = ActivationState.DisableDeactivation;
		}
		rbInfo.Dispose();

		m_world.AddRigidBody(body);

		return body;
	}

	protected void AddRigidBody(RigidBody body) {
		m_world.AddRigidBody(body);
	}

	// Removes the given rigid body from the world and cleans up after it.
	protected void DestroyRigidBody(CollisionObject obj) {
		RigidBody body = obj as RigidBody;
		if (body != null && body.MotionState != null) {
			body.MotionState.Dispose();
		}
		m_world.RemoveCollisionObject(obj);
		obj.Dispose();
	}

	protected void RemoveRigidBody(RigidBody obj) {
		m_world.RemoveCollisionObject(obj);
	}

	protected GameObject GetLastUnityObject() {
		return m_createdObjs[m_createdObjs.Count - 1];
	}

	// destroys the most recently added unity object
	protected void DestoryLastUnityObject() {
		GameObject lastObj = m_createdObjs[m_createdObjs.Count - 1];
		m_createdObjs.RemoveAt(m_createdObjs.Count - 1);
		var meshFilter = lastObj.GetComponent<MeshFilter>();
		Destroy(meshFilter.mesh);
		Destroy(lastObj);
	}

	// Creates a Unity game object from the given Bullet CollisionObject.
	protected void AddUnityObject(CollisionObject co, Material mat) {
		CollisionShape cs = co.CollisionShape;
		GameObject go;
		if (cs.ShapeType == BroadphaseNativeType.SoftBodyShape) {
			BulletSharp.SoftBody.SoftBody sb = (BulletSharp.SoftBody.SoftBody)co;
			if (sb.Faces.Count == 0) {
				//rope
				go = CreateUnitySoftBodyRope(sb);
			} else {
				go = CreateUnitySoftBodyCloth(sb);
			}
		} else {
			//rigid body
			if (cs.ShapeType == BroadphaseNativeType.CompoundShape) {
				//BulletSharp.Math.Matrix transform = co.WorldTransform;
				go = new GameObject("Compund Shape");
				BulletRigidBodyProxy rbp = go.AddComponent<BulletRigidBodyProxy>();
				rbp.target = co as RigidBody;
				foreach (BulletSharp.CompoundShapeChild child in (cs as CompoundShape).ChildList) {
					BulletSharp.Math.Matrix childTransform = child.Transform;
					GameObject ggo = new GameObject(child.ToString());
					MeshFilter mf = ggo.AddComponent<MeshFilter>();
					Mesh m = mf.mesh;
					MeshFactory2.CreateShape(child.ChildShape, m);
					MeshRenderer mr = ggo.AddComponent<MeshRenderer>();
					mr.sharedMaterial = mat;
					ggo.transform.SetParent(go.transform);
					Matrix4x4 mt = childTransform.ToUnity();
					ggo.transform.localPosition = BSExtensionMethods2.ExtractTranslationFromMatrix(ref mt);
					ggo.transform.localRotation = BSExtensionMethods2.ExtractRotationFromMatrix(ref mt);
					ggo.transform.localScale = BSExtensionMethods2.ExtractScaleFromMatrix(ref mt);

					/*
                        BulletRigidBodyProxy rbp = ggo.AddComponent<BulletRigidBodyProxy>();
                        rbp.target = body;
                        return go;
                        */
					//InitRigidBodyInstance(colObj, child.ChildShape, ref childTransform);
				}
			} else if (cs.ShapeType == BroadphaseNativeType.CapsuleShape) {
				CapsuleShape css = (CapsuleShape) cs;
				GameObject ggo = GameObject.CreatePrimitive(PrimitiveType.Capsule);
				Destroy(ggo.GetComponent<Collider>());
				go = new GameObject();
				ggo.transform.parent = go.transform;
				ggo.transform.localPosition = UnityEngine.Vector3.zero;
				ggo.transform.localRotation = UnityEngine.Quaternion.identity;
				ggo.transform.localScale = new UnityEngine.Vector3(css.Radius * 2f,css.HalfHeight * 2f,css.Radius * 2f);
				BulletRigidBodyProxy rbp = go.AddComponent<BulletRigidBodyProxy>();
				rbp.target = co;
			} else { 
				//Debug.Log("Creating " + cs.ShapeType + " for " + co.ToString());
				go = CreateUnityCollisionObjectProxy(co as CollisionObject, mat);
			}
		}
		m_createdObjs.Add(go);
	}


	protected GameObject CreateUnityCollisionObjectProxy(CollisionObject body, Material mat) {
		if (body is GhostObject)
		{
			Debug.Log("ghost obj");
		}
		GameObject go = new GameObject(body.ToString());
		MeshFilter mf = go.AddComponent<MeshFilter>();
		Mesh m = mf.mesh;
		MeshFactory2.CreateShape(body.CollisionShape, m);
		MeshRenderer mr = go.AddComponent<MeshRenderer>();
		mr.sharedMaterial = mat;
		BulletRigidBodyProxy rbp = go.AddComponent<BulletRigidBodyProxy>();
		rbp.target = body;
		return go;
	}

	protected GameObject CreateUnitySoftBodyRope(BulletSharp.SoftBody.SoftBody body) {
		//determine what kind of soft body it is
		//rope
		GameObject rope = Instantiate<GameObject>(ropePrefab);
		LineRenderer lr = rope.GetComponent<LineRenderer>();
		lr.SetVertexCount(body.Nodes.Count);
		BulletRopeProxy ropeProxy = rope.GetComponent<BulletRopeProxy>();
		ropeProxy.target = body;
		return rope;
	}

	protected GameObject CreateUnitySoftBodyCloth(BulletSharp.SoftBody.SoftBody body) {
		//build nodes 2 verts map
		Dictionary<BulletSharp.SoftBody.Node, int> node2vertIdx = new Dictionary<BulletSharp.SoftBody.Node, int>();
		for (int i = 0; i < body.Nodes.Count; i++) {
			node2vertIdx.Add(body.Nodes[i], i);
		}
		List<int> tris = new List<int>();
		for (int i = 0; i < body.Faces.Count; i++) {
			BulletSharp.SoftBody.Face f = body.Faces[i];
			if (f.Nodes.Count != 3) {
				Debug.LogError("Face was not a triangle");
				continue;
			}
			for (int j = 0; j < f.Nodes.Count; j++) { 
				tris.Add( node2vertIdx[f.Nodes[j]]);
			}
		}
		GameObject go = Instantiate<GameObject>(softBodyPrefab);
		BulletSoftBodyProxy sbp = go.GetComponent<BulletSoftBodyProxy>();
		List<int> trisRev = new List<int>();
		for (int i = 0; i < tris.Count; i+=3) {
			trisRev.Add(tris[i]);
			trisRev.Add(tris[i + 2]);
			trisRev.Add(tris[i + 1]);
		}
		tris.AddRange(trisRev);
		sbp.target = body;
		sbp.verts = new UnityEngine.Vector3[body.Nodes.Count];
		sbp.tris = tris.ToArray();
		return go;
	}

	protected void SetEyeTarget(BulletSharp.Math.Vector3 eye, BulletSharp.Math.Vector3 targ) {
		UnityEngine.Transform t = UnityEngine.Camera.main.transform;
		t.position = eye.ToUnity();
		t.rotation = UnityEngine.Quaternion.LookRotation((targ - eye).ToUnity().normalized, UnityEngine.Vector3.up);

	}

	// Disposes of all Unity and Bullet objects and closes the application
	protected void ExitSimulation() {
		Debug.Log("Quitting simulation...");
		m_quitting = true;
		// destroy all Unity objects
		for (int i = 0; i < m_createdObjs.Count; i++) {
			Destroy(m_createdObjs[i]);
		}
		m_createdObjs.Clear();

		// destroy all bullet objects
		if (m_world != null)
		{
			//remove/dispose constraints
			int i;
			for (i = m_world.NumConstraints - 1; i >= 0; i--)
			{
				TypedConstraint constraint = m_world.GetConstraint(i);
				m_world.RemoveConstraint(constraint);
				constraint.Dispose();
			}

			//remove the rigidbodies from the dynamics world and delete them
			for (i = m_world.NumCollisionObjects - 1; i >= 0; i--)
			{
				CollisionObject obj = m_world.CollisionObjectArray[i];
				RigidBody body = obj as RigidBody;
				if (body != null && body.MotionState != null)
				{
					body.MotionState.Dispose();
				}
				m_world.RemoveCollisionObject(obj);
				obj.Dispose();
			}

			//delete collision shapes
			foreach (CollisionShape shape in m_collisionShapes)
				shape.Dispose();
			m_collisionShapes.Clear();

			m_world.Dispose();
			m_broadphase.Dispose();
			m_colDispatcher.Dispose();
			m_colConfig.Dispose();
		}

		if (m_broadphase != null)
		{
			m_broadphase.Dispose();
		}
		if (m_colDispatcher != null)
		{
			m_colDispatcher.Dispose();
		}
		if (m_colConfig != null)
		{
			m_colConfig.Dispose();
		}

		Application.Quit();
	}

	//
	// METHODS to be overriden in derived classes
	//

	// Sets up initial collision shapes and objects for the simulation.
	protected virtual void InitSim() {
		SetTimeScale (3.0f);

		BulletSharp.Math.Vector3 eye = new BulletSharp.Math.Vector3(30, 20, 10);
		BulletSharp.Math.Vector3 target = new BulletSharp.Math.Vector3(0, 5, -4);

		// create 125 (5x5x5) dynamic objects
		const int ArraySizeX = 5, ArraySizeY = 5, ArraySizeZ = 5;

		// scaling of the objects (0.1 = 20 centimeter boxes )
		const float StartPosX = -5;
		const float StartPosY = -5;
		const float StartPosZ = -3;

		SetEyeTarget(eye, target);

		// create the ground
		BoxShape groundShape = new BoxShape(50, 1, 50);
		AddCollisionShape(groundShape);

		Material mat = new Material(Shader.Find("Diffuse"));
		CollisionObject ground = CreateRigidBody(0, Matrix.Identity, groundShape, mat);
		ground.UserObject = "Ground";

		// create a few dynamic rigidbodies
		const float mass = 1.0f;

		BoxShape colShape = new BoxShape(1);
		AddCollisionShape(colShape);

		const float startX = StartPosX - ArraySizeX / 2;
		const float startY = StartPosY;
		const float startZ = StartPosZ - ArraySizeZ / 2;

		for (int k = 0; k < ArraySizeY; k++)
		{
			for (int i = 0; i < ArraySizeX; i++)
			{
				for (int j = 0; j < ArraySizeZ; j++)
				{
					Matrix startTransform = Matrix.Translation(
						2 * i + startX,
						2 * k + startY,
						2 * j + startZ
					);

					RigidBody body = CreateRigidBody (mass, startTransform, colShape, mat);
					// make it drop from a height
					body.Translate(new BulletSharp.Math.Vector3(0, 20, 0));
				}
			}
		}

	}

	// Called after each simulation step to perform any necessary updates.
	protected virtual void UpdateSim() {

	}

	protected virtual void RenderUpdateSim() {
		
	}

}
