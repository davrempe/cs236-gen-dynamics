using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

using BulletSharp;
using BulletSharp.Math;
using BulletUnity;

public class StepSimpleDataGen : BulletSimBase {

	//
	// Publicly exposed members (in the editor).
	//

	public bool BUILD_MODE = false;

	// directory in Resources where .obj files of objects to simulate are located
	public string dataInDir = "SimDataModels"; 
	// directory where we out simulation data
	public string dataOutDir = "./BulletDataOut/";
	public int startingObjIdx = 0;
	// number of simulations to run for each loaded .obj file
	public int numSimsPerObject = 1;

	// will dump out object state every this number of frames
    public int saveFrequency = 10;

	public bool perturb = false;
	public int perturbFrequency = 70;
	public int maxPerturbations = 4;
	public float linearStrength = 0.2f;
	public float angularStrength = 0.2f;

	// applied impulse range
	public float impulseMin = 0.0f;
	public float impulseMax = 10.0f;
	public float forceDistMax = 10.0f;

	// the max mass to normalize to
	// public float massMax = 1.5f;

	// if true, chooses the minimum density for all shapes
	public bool useConstantDensity = true;
	// density range to choose from while simulating (in kg/m^3)
	public float densityMin = 10;
	public float densityMax = 50;

	// physics settings
	public float groundFriction = 0.5f;
	public float bodyFriction = 0.5f;
	public float bodyMargin = 0.05f;
	public BulletSharp.Math.Vector3 angularFactor = new BulletSharp.Math.Vector3(1.0f);
	public float linearSleepThresh = 0.8f;
	public float angularSleepThresh = 1.0f;

	// random scale params
	public bool randomScale = false;
	public bool varyScale = false; // vary the scale differently for each dimension.
    public float scaleMin = 0.25f;
    public float scaleMax = 3.0f;

	// camera settings
	public BulletSharp.Math.Vector3 cameraLoc = new BulletSharp.Math.Vector3(30, 20, 10);
	public BulletSharp.Math.Vector3 cameraTarget = new BulletSharp.Math.Vector3(0, 5, -4);

	// object appearance
	public Material groundMat;
	public Material bodyMat;

	// if not -1, will only simulate the idx loaded
	public bool DEBUG = true;
	public int DEBUG_simFileIdx = -1;
	public bool RENDER_MODE = true;
	public bool TRACK_FALLS = false;
	public int numFallOrStuckLimit = 7;
	public bool SAVE_VIDEO = false;
	public string videoOutDir = "./SimVideos/";

	//
	// Internal members
	//
    
	protected List<UnityEngine.Vector3> m_stepVel;
	protected List<UnityEngine.Vector3> m_stepAngVel;
	protected List<UnityEngine.Quaternion> m_stepRot;
	protected List<UnityEngine.Vector3> m_stepEulerRot;
	protected List<UnityEngine.Vector3> m_stepPos;
	protected List<int> m_stepNum;

	protected List<string> m_fallenFiles;

	// list of all the obj files to sim
	protected string[] m_objFiles;
	// base directory containing all meshes to read in
	protected string m_baseDataDir;
	// the masses of the objects in the files
	protected float[] m_masses;
	// the moment of inertial tensors of the object in the files
	protected List<BulletSharp.Math.Vector3> m_inertias;
	// cur idx of simulating obj
	protected int m_curObjIdx;
	// the current output data dir
	protected string m_curDataOutDir;
	// save video
	protected string m_curVideoOutDir;
	// cur simulation number for this object
	protected int m_curSimNum;
	// the static rigid body for the ground
	protected RigidBody m_groundRb;
	// starting position of the ground
	protected Matrix m_groundInitTrans;
	// collision margin on ground
	protected float m_groundMargin;
	// currently simulated rigid body
	protected RigidBody m_rb;
	// starting position for currently simulating rigid body
	protected Matrix m_rbInitTrans;
	protected BulletSharp.Math.Vector3 m_rbInitTransVec;
	// the current canonical bullet mesh for the collision shape
	DataGenUtils.BulletOBJMesh m_btmesh;
	// currently simulated collision shape
	protected GImpactMeshShape m_cs;

	protected int m_fallCount;
	protected int m_stepCount;
	protected int m_stuckCount;

	protected float m_prevTime;

	protected int m_renderFrame;
	protected int m_numPerturbs;

	//
	// Data to save
	//
	protected float m_density;
	protected float m_mass;
	protected BulletSharp.Math.Vector3 m_inertia;
	protected UnityEngine.Vector3 m_pclScale; // the scale of the point cloud in each dimension
	protected BulletSharp.Math.Vector3 m_forcePoint;
	protected BulletSharp.Math.Vector3 m_forceVec;
	protected BulletSharp.Math.Vector3 m_totalRot;
	protected BulletSharp.Math.Vector3 m_vel0; // initial linear velocity after impulse
	protected BulletSharp.Math.Vector3 m_angvel0; // initial angulr velocity after impulse
	protected UnityEngine.Vector3 m_pos0; // object pos (x, y, z) at first frame
	protected BulletSharp.Math.Vector3 m_com0; // COM pos (x, y, z) at first frame
	protected UnityEngine.Quaternion m_rot0; // object rotation (w, x, y, z) at first frame
	protected UnityEngine.Vector3 m_eulerrot0; // euler rotation first frame
	protected UnityEngine.Vector3 m_posf; // last frame
	protected BulletSharp.Math.Vector3 m_comf; // last frame com
	protected UnityEngine.Quaternion m_rotf; // last frame
	protected UnityEngine.Vector3 m_eulerrotf; // euler rotation last frame

	// Sets up initial collision shapes and objects for the simulation.
	protected override void InitSim() {
		Debug.Log ("InitSim");

		// create data directory if doesn't exist
		Directory.CreateDirectory(dataOutDir);

		// get list of all objs to sim
		if (!BUILD_MODE) {
			m_baseDataDir = Application.dataPath + "/Resources/" + dataInDir + "/";
		} else {
			m_baseDataDir = Application.dataPath + "/../../Assets/Resources/" + dataInDir + "/";
		}
		DirectoryInfo resDirPath = new DirectoryInfo(m_baseDataDir);
		FileInfo[] objFileInfos = resDirPath.GetFiles("*.obj", SearchOption.AllDirectories);
		Debug.Log("Files to sim: " + objFileInfos.Length.ToString());
		m_objFiles = new string[objFileInfos.Length];
		int fileIdx = 0;
		foreach (FileInfo file in objFileInfos) {
			m_objFiles[fileIdx++] = file.Name;
		}

		// First read in all volumes
		m_masses = new float[m_objFiles.Length];
		float maxVol = -float.MaxValue;
		m_inertias = new List<BulletSharp.Math.Vector3>();
		for (int i = 0; i < m_masses.Length; i++) {
			string infoFile = dataInDir + "/" + m_objFiles[i].Replace(".obj", "");
			/*if (DEBUG)*/ Debug.Log("Idx " + i.ToString() + ": " + infoFile);
			var jsonTextFile = Resources.Load<TextAsset>(infoFile);
			if (DEBUG) Debug.Log(jsonTextFile.ToString());
			ObjInfo curInfo = ObjInfo.CreateFromJSON(jsonTextFile.ToString());
			m_masses[i] = curInfo.vol;
            // moment of inertias assume the density is 1 (i.e. mass = volume)
			m_inertias.Add(new BulletSharp.Math.Vector3(curInfo.inertia[0], curInfo.inertia[1], curInfo.inertia[2]));
			if (DEBUG) Debug.Log(m_masses[i].ToString());
			if (m_masses[i] > maxVol) {
				maxVol = m_masses[i];
			}
		}

		// Normalize volumes and use as mass
		// float normCoeff = massMax / maxVol; // need to scale density by this much.
		// for (int i = 0; i < m_masses.Length; i++) {
		// 	m_masses[i] = normCoeff * m_masses[i];
		// 	m_inertias[i] = normCoeff * m_inertias[i];
		// 	if (DEBUG) Debug.Log(m_masses[i].ToString());
		// 	if (DEBUG) Debug.Log(m_inertias[i].ToString());
		// }

		// only simulate one of interest for debugging
		if (DEBUG_simFileIdx != -1) {
			string simObjFile = m_objFiles[DEBUG_simFileIdx];
			float simMass = m_masses[DEBUG_simFileIdx];
			BulletSharp.Math.Vector3 simInertia = m_inertias[DEBUG_simFileIdx];
			m_objFiles = new string[1];
			m_objFiles[0] = simObjFile;
			m_masses = new float[1];
			m_masses[0] = simMass;
			m_inertias = new List<BulletSharp.Math.Vector3>();
			m_inertias.Add(simInertia);
			Debug.Log("ONLY SIMULATING: " + simObjFile);
		}

		// create the ground, this will be the same for every sim
		BulletSharp.Math.Vector3 groundSize = new BulletSharp.Math.Vector3(50, 0.04f, 50);
		BoxShape groundShape = new BoxShape(groundSize);
		// groundShape.Margin = 0.07f;
		AddCollisionShape(groundShape);
		m_groundMargin = groundShape.Margin;
		// move down so surface is exactly y=0
		Debug.Log("GROUND MARGIN: " + m_groundMargin.ToString());
		// The 0.016 is dependent on the ground margin but idk why at this point
		m_groundInitTrans = Matrix.Translation(new BulletSharp.Math.Vector3(0, -((groundSize[1] / 2.0f) + 0.016f), 0));
		m_groundRb = CreateRigidBody(0, m_groundInitTrans, groundShape, groundMat, groundFriction, viz : RENDER_MODE);

		// setup the camera
		SetEyeTarget(cameraLoc, cameraTarget);

		// Now load the first simulation object to start
		m_curObjIdx = startingObjIdx;
		if (m_objFiles.Length == 0) {
			ExitSimulation();
		} else {
			m_curDataOutDir = Path.Combine(dataOutDir, m_objFiles[m_curObjIdx].Replace(".obj", "") + "/");
			Directory.CreateDirectory(m_curDataOutDir);	
			PrepareSimObj(m_baseDataDir + m_objFiles[m_curObjIdx], m_masses[m_curObjIdx], m_inertias[m_curObjIdx]);

			m_curVideoOutDir = Path.Combine(Application.dataPath, videoOutDir);
			m_curVideoOutDir = Path.Combine(m_curVideoOutDir, m_objFiles[m_curObjIdx].Replace(".obj", "") + "/");
			Directory.CreateDirectory(m_curVideoOutDir);
			string simOutDir = Path.Combine(m_curVideoOutDir, "sim_0");
			if (!Directory.Exists(simOutDir)) Directory.CreateDirectory(simOutDir);
		}
	
		// so we can handle object collisions, only need to do this once
		GImpactCollisionAlgorithm.RegisterAlgorithm(m_colDispatcher);

		InitStateLists();

		// determine and apply the force for the first simulation
		SetupSim();
		// take note of inital rigid body state
		SaveInitState();
		SaveCurrentState();
		m_curSimNum = 0;
		m_renderFrame = 1;
		// set up rotation tracking
		m_totalRot = new BulletSharp.Math.Vector3(0.0f, 0.0f, 0.0f);

		m_pclScale = UnityEngine.Vector3.one;

		m_fallenFiles = new List<string>();
		m_fallCount = 0;
		m_stepCount = 0;
		m_stuckCount = 0;
		m_numPerturbs = 0;
	}

	//public Material lineMat = new Material("Shader \"Lines/Colored Blended\" {" + "SubShader { Pass { " + "    Blend SrcAlpha OneMinusSrcAlpha " + "    ZWrite Off Cull Off Fog { Mode Off } " + "    BindChannels {" + "      Bind \"vertex\", vertex Bind \"color\", color }" + "} } }");

	//void OnPostRender()
	//{
	//    GL.Begin(GL.LINES);
	//    lineMat.SetPass(0);
	//    GL.Color(new Color(0f, 0f, 0f, 1f));
	//    GL.Vertex3(0f, 0f, 0f);
	//    GL.Vertex3(1f, 1f, 1f);
	//    GL.End();
	//}
    
	protected override void RenderUpdateSim()
	{
		base.RenderUpdateSim();

		if (SAVE_VIDEO)
        {
            string outFile = Path.Combine(m_curVideoOutDir, "sim_" + m_curSimNum.ToString());
			string frameNum = m_renderFrame.ToString().PadLeft(6, '0');
			outFile = Path.Combine(outFile, "frame_" + frameNum + ".png");
            //string outFile = m_curVideoOutDir + "sim_" + m_curSimNum.ToString() + "/frame_" + m_stepCount;
            ScreenCapture.CaptureScreenshot(outFile, 1);
            //Texture2D screen = ScreenCapture.CaptureScreenshotAsTexture();
            //byte[] bytes = screen.EncodeToPNG();
            //File.WriteAllBytes(outFile, bytes);
            Debug.Log(outFile);
			//while(!File.Exists(outFile + ".png")) {}
			m_renderFrame++;
        }
	}

	// Called after each simulation step to perform any necessary updates.
	protected override void UpdateSim() {
//		Debug.Log ("UpdateSim");
		//if (DEBUG) Debug.DrawRay(BSExtensionMethods2.ToUnity(m_rb.CenterOfMassPosition), BSExtensionMethods2.ToUnity(m_rb.LinearVelocity), Color.red, 20.0f);
		if (DEBUG) Debug.DrawRay(BSExtensionMethods2.ToUnity(m_rb.CenterOfMassPosition), (BSExtensionMethods2.ToUnity(m_rb.LinearVelocity) / m_rb.LinearVelocity.Length)*0.01f, Color.red, 20.0f);


		//if (m_stepCount == 0) {
		//	Debug.Log("INIT LINVEL2: " + m_rb.LinearVelocity.ToString());
		//	Debug.Log("INIT ANGVEL2: " + m_rb.AngularVelocity.ToString());
		//}

		// update rotation tracker
		BulletSharp.Math.Vector3 angleRotated = Time.fixedDeltaTime * (Mathf.Rad2Deg * m_rb.AngularVelocity);
		//if ((m_totalRot[1] > 0 && angleRotated[1] < 0) || (m_totalRot[1] < 0 && angleRotated[1] > 0)) {
		//	Debug.Log("SWAP: " + angleRotated[1].ToString());
		//}
		m_totalRot += angleRotated;

		// check if the current rigid body is still moving (isactive)
		if (!m_rb.IsActive) {
			if (DEBUG) Debug.Log("TOTAL ROT: " + m_totalRot.ToString());
			// record the current state and dump all info to file
			SaveCurrentState();
			SaveFinalState();
			DumpSimInfo();
			// check if we're done with this object;
			if (++m_curSimNum >= numSimsPerObject) {
				if (++m_curObjIdx >= m_objFiles.Length) {
					Debug.Log("BAD FILES: " + m_fallenFiles.Count.ToString() + ", NUM STUCK: " + m_stuckCount.ToString());
					for (int k = 0; k < m_fallenFiles.Count; k++) {
						Debug.Log(m_fallenFiles[k]);
					}
					DumpFallenObjects();
					ExitSimulation();
				} else {
					DestroySimObj();
					// create new directory to store sims from the next object
					m_curDataOutDir = Path.Combine(dataOutDir, m_objFiles[m_curObjIdx].Replace(".obj", "") + "/");
					Directory.CreateDirectory(m_curDataOutDir);
					PrepareSimObj(m_baseDataDir + m_objFiles[m_curObjIdx], m_masses[m_curObjIdx], m_inertias[m_curObjIdx]);

					m_curVideoOutDir = Path.Combine(Application.dataPath, videoOutDir);
					m_curVideoOutDir = Path.Combine(m_curVideoOutDir, m_objFiles[m_curObjIdx].Replace(".obj", "") + "/");
					Directory.CreateDirectory(m_curVideoOutDir);
					string curSimOut = Path.Combine(m_curVideoOutDir, "sim_0");
					if (!Directory.Exists(curSimOut)) Directory.CreateDirectory(curSimOut);
					m_renderFrame = 1;

					// ready the first sim
					SetupSim();
					SaveInitState();
					ClearStateLists();
					m_curSimNum = 0;
					// reset rotation tracking
					m_totalRot = new BulletSharp.Math.Vector3(0.0f, 0.0f, 0.0f);
					m_numPerturbs = 0;
				}
				m_fallCount = 0;
			} else {
				// set up next sim with the same object
				SetupSim();
				SaveInitState();
				ClearStateLists();
				// reset rotation tracking
				m_totalRot = new BulletSharp.Math.Vector3(0.0f, 0.0f, 0.0f);
				m_numPerturbs = 0;

				string curSimOut = Path.Combine(m_curVideoOutDir, "sim_" + m_curSimNum.ToString());
				if (!Directory.Exists(curSimOut)) Directory.CreateDirectory(curSimOut);
				m_renderFrame = 1;
			}
		} else if (Mathf.Abs(m_totalRot[0]) > 20.0f || Mathf.Abs(m_totalRot[2]) > 20.0f || (++m_stepCount > (1.0 / simStepSize) * 7.0f)) {
			if (DEBUG) Debug.Log("FELL OVER OR GOT STUCK...RESTARTING CURRENT SIM.");
			if (TRACK_FALLS)
			{
				m_fallCount++;
				// // keep track of fallen ones in a list and move on to next object
				if (m_fallCount >= numFallOrStuckLimit)
				{
					if (DEBUG) Debug.Log("REACHED FALL/STUCK LIMIT, MOVING TO NEXT OBJECT.");
					m_fallenFiles.Add(m_objFiles[m_curObjIdx]);
					if (++m_curObjIdx >= m_objFiles.Length)
					{
						Debug.Log("BAD FILES: " + m_fallenFiles.Count.ToString() + ", NUM STUCK: " + m_stuckCount.ToString());
						for (int k = 0; k < m_fallenFiles.Count; k++)
						{
							Debug.Log(m_fallenFiles[k]);
						}
						DumpFallenObjects();
						ExitSimulation();
					}
					else
					{
						DestroySimObj();
						// create new directory to store sims from the next object
						m_curDataOutDir = Path.Combine(dataOutDir, m_objFiles[m_curObjIdx].Replace(".obj", "") + "/");
						Directory.CreateDirectory(m_curDataOutDir);
						PrepareSimObj(m_baseDataDir + m_objFiles[m_curObjIdx], m_masses[m_curObjIdx], m_inertias[m_curObjIdx]);

						m_curVideoOutDir = Path.Combine(Application.dataPath, videoOutDir);
						m_curVideoOutDir = Path.Combine(m_curVideoOutDir, m_objFiles[m_curObjIdx].Replace(".obj", "") + "/");
                        Directory.CreateDirectory(m_curVideoOutDir);
						string curSimOut = Path.Combine(m_curVideoOutDir, "sim_0");
                        Directory.CreateDirectory(curSimOut);
						m_renderFrame = 1;

						// ready the first sim
						SetupSim();
						SaveInitState();
						ClearStateLists();
						m_curSimNum = 0;
						// reset rotation tracking
						m_totalRot = new BulletSharp.Math.Vector3(0.0f, 0.0f, 0.0f);
						m_numPerturbs = 0;
					}
					m_fallCount = 0;
				}
			}

			// the object has probably fallen over, restart
			SetupSim();
			SaveInitState();
			ClearStateLists();
			// reset rotation tracking
			m_totalRot = new BulletSharp.Math.Vector3(0.0f, 0.0f, 0.0f);
			m_numPerturbs = 0;
		// } else if (++m_stepCount > (1.0 / simStepSize) * 20) { // been simulating for more than 10 "seconds" (jittering)
		// 	m_stuckCount++;
		// 	if (DEBUG) Debug.Log("SIMULATION STUCK, MOVING TO NEXT OBJECT");
		// 	m_fallenFiles.Add(m_objFiles[m_curObjIdx]);
		// 	if (++m_curObjIdx >= m_objFiles.Length) {
		// 		Debug.Log("BAD FILES: " + m_fallenFiles.Count.ToString() + ", NUM STUCK: " + m_stuckCount.ToString());
		// 		for (int k = 0; k < m_fallenFiles.Count; k++) {
		// 			Debug.Log(m_fallenFiles[k]);
		// 		}
		// 		DumpFallenObjects();
		// 		ExitSimulation();
		// 	} else {
		// 		DestroySimObj();
		// 		// create new directory to store sims from the next object
		// 		m_curDataOutDir = dataOutDir + m_objFiles[m_curObjIdx].Replace(".obj", "") + "/";
		// 		Directory.CreateDirectory(m_curDataOutDir);
		// 		PrepareSimObj(m_baseDataDir + m_objFiles[m_curObjIdx], m_masses[m_curObjIdx], m_inertias[m_curObjIdx]);
		// 		// ready the first sim
		// 		SetupSim();
		// 		SaveInitState();
		// 		m_curSimNum = 0;
		// 		// reset rotation tracking
		// 		m_totalRot = new BulletSharp.Math.Vector3(0.0f, 0.0f, 0.0f);
		// 	}

		// 	m_fallCount = 0;

		// 	// the object has probably fallen over, restart
		// 	SetupSim();
		// 	SaveInitState();
		// 	// reset rotation tracking
		// 	m_totalRot = new BulletSharp.Math.Vector3(0.0f, 0.0f, 0.0f);
		}

		if (m_stepCount % saveFrequency == 0) {
            SaveCurrentState();
        }

		if (perturb && m_stepCount != 0 && m_stepCount % perturbFrequency == 0 && m_numPerturbs < maxPerturbations) {
			if (DEBUG) Debug.Log("PERTURBED!");
            // randomly perturb velocities
			Vector2 linPerturb = linearStrength * Random.insideUnitCircle;
			m_rb.LinearVelocity += new BulletSharp.Math.Vector3(linPerturb[0], 0.0f, linPerturb[1]);
            
			float angPerturb = angularStrength * Random.Range(0.0f, 1.0f);
			m_rb.AngularVelocity += new BulletSharp.Math.Vector3(0.0f, angPerturb, 0.0f);

			m_numPerturbs++;
		}

		//if (DEBUG) Debug.Log("WORLD MOMENT: " + m_rb.InvInertiaTensorWorld.ToString());
		//if (DEBUG) Debug.Log("WORLD BASIS: " + m_rb.WorldTransform.Basis.ToString());
	}

	// Loads the given OBJ file, creates a RigidBody with the given mess, and places
	// at the origin of the ground.
	protected void PrepareSimObj(string objFile, float mass, BulletSharp.Math.Vector3 inertia) {
		Debug.Log("Loading " + objFile + "...");
		// Load wavefront file
		OBJLoader.OBJMesh objloadermesh = OBJLoader.LoadOBJMesh(objFile);
		Debug.Assert(objloadermesh.vertices.Count > 0);
		Debug.Assert(objloadermesh.faces.Count > 0);
		//		Debug.Log("VERTS: " + objloadermesh.vertices.Count.ToString());
		//		Debug.Log("FACES: " + objloadermesh.faces.Count.ToString());

		m_btmesh = DataGenUtils.BulletMeshFromUnity(objloadermesh);
		Debug.Assert(m_btmesh.vertices.Length > 0);
		Debug.Assert(m_btmesh.indices.Length > 0);
		//		Debug.Log("btVERTS: " + (btmesh.vertices.Length / 3).ToString());
		//		Debug.Log("btFACES: " + (btmesh.indices.Length / 3).ToString());

		// Create a GImpactMeshShape for collider
		var triVtxarray = new TriangleIndexVertexArray(m_btmesh.indices, m_btmesh.vertices);
		m_cs = new GImpactMeshShape(triVtxarray);
		m_cs.LocalScaling = new BulletSharp.Math.Vector3(1);
		m_cs.Margin = bodyMargin;
		m_cs.UpdateBound();
		AddCollisionShape(m_cs);

		// move it up so resting on the ground plane
		float miny = float.MaxValue;
		float cury;
		for (int i = 0; i < objloadermesh.vertices.Count; i++) {
			cury = objloadermesh.vertices[i][1];
			if (cury < miny) {
				miny = cury;
			}
		}
		miny = -miny;
		m_rbInitTransVec = new BulletSharp.Math.Vector3(0, miny + bodyMargin + m_groundMargin, 0);
		m_rbInitTrans = Matrix.Translation(m_rbInitTransVec);// * Matrix.RotationY(Random.Range(0.0f, 360.0f));
		m_rb = CreateRigidBody(mass, inertia, m_rbInitTrans, m_cs, bodyMat, bodyFriction, viz: RENDER_MODE);
		m_rb.AngularFactor = angularFactor;
		m_rb.SetSleepingThresholds(linearSleepThresh, angularSleepThresh);
		if (DEBUG) Debug.Log("LOADED MOMENT: " + m_rb.LocalInertia.ToString());
		// if (DEBUG) Debug.Log("WORLD MOMENT: " + m_rb.InvInertiaTensorWorld.ToString());
//		Debug.Log("Min y: " + (-miny).ToString());
		if (DEBUG) Debug.Log(m_rb.CenterOfMassPosition.ToString());


	}

	protected void DestroySimObj() {
		// remove the current collision shape and rigid body
		DestroyRigidBody(m_rb);
		RemoveCollisionShape(m_cs);
		if (RENDER_MODE) DestoryLastUnityObject();
	}

	protected void SetupSim() {
		// randomly choose scaling before resetting rigid body
		float randomMass = 0;
		BulletSharp.Math.Vector3 randomInertia;
		if (randomScale) {
			if (varyScale) {
				m_pclScale.x = Random.Range(this.scaleMin, this.scaleMax);
				m_pclScale.y = Random.Range(this.scaleMin, this.scaleMax);
				m_pclScale.z = Random.Range(this.scaleMin, this.scaleMax);
			} else {
				float uniformScale = Random.Range(this.scaleMin, this.scaleMax);
				m_pclScale.x = uniformScale;
				m_pclScale.y = uniformScale;
				m_pclScale.z = uniformScale;
			}
			//// z can't be more than thrice or less than half of x scale
			//float zmin = Mathf.Max(this.scaleMin, 0.5f * m_pclScale.x);
			//float zmax = Mathf.Min(this.scaleMax, 3.0f * m_pclScale.x);
			//randomScale = Random.Range(zmin, zmax);
			//m_pclScale.z = randomScale;
			//// y can't be greater than 2 times the smallest of x and z
			//float ymax = 2.0f * Mathf.Min(m_pclScale.x, m_pclScale.z);
			//randomScale = Random.Range(this.scaleMin, Mathf.Min(ymax, this.scaleMax));
			//m_pclScale.y = randomScale;

			if (DEBUG) Debug.Log("Scaling by " + m_pclScale.ToString());

			// randomMass = m_masses[m_curObjIdx] * m_pclScale.x * m_pclScale.y * m_pclScale.z;
			float randomDensity;
			if (useConstantDensity) {
				// density is constant so mass must scale with volume
				randomDensity = densityMin;
				randomMass = randomDensity * m_masses[m_curObjIdx] * m_pclScale.x * m_pclScale.y * m_pclScale.z;
			} else {
				randomDensity = Random.Range(densityMin, densityMax);
				randomMass = randomDensity * m_masses[m_curObjIdx];
			}
            // inertia must scale with volume no matter if the density is constant or not
			BulletSharp.Math.Vector3 objInertiaInfo = m_inertias[m_curObjIdx];
            float scalexyz = m_pclScale.x * m_pclScale.y * m_pclScale.z;
            float scalex2 = m_pclScale.x * m_pclScale.x;
            float scaley2 = m_pclScale.y * m_pclScale.y;
            float scalez2 = m_pclScale.z * m_pclScale.z;
            float inertiax = randomDensity * scalexyz * (scaley2 * objInertiaInfo[1] + scalez2 * objInertiaInfo[2]);
            float inertiay = randomDensity * scalexyz * (scalex2 * objInertiaInfo[0] + scalez2 * objInertiaInfo[2]);
            float inertiaz = randomDensity * scalexyz * (scalex2 * objInertiaInfo[0] + scaley2 * objInertiaInfo[1]);
            randomInertia = new BulletSharp.Math.Vector3(inertiax, inertiay, inertiaz);

			// need to completely destory rigid body because need new mass/moment of inertia
			DestroySimObj();

			DataGenUtils.BulletOBJMesh scaledMesh = m_btmesh.Scale(m_pclScale.x, m_pclScale.y, m_pclScale.z);
			var triVtxarray = new TriangleIndexVertexArray(scaledMesh.indices, scaledMesh.vertices);
			m_cs = new GImpactMeshShape(triVtxarray);
			m_cs.LocalScaling = new BulletSharp.Math.Vector3(1);
			m_cs.Margin = bodyMargin;
			m_cs.UpdateBound();
			AddCollisionShape(m_cs);

			// move it up so resting on the ground plane
			float miny = float.MaxValue;
			float maxz = float.MinValue;
			float cury;
			float curz;
			for (int i = 0; i < scaledMesh.vertices.Length / 3; i++) {
				cury = scaledMesh.vertices[i*3 + 1];
				if (cury < miny) {
					miny = cury;
				}
				curz = scaledMesh.vertices[i * 3 + 2];
				if (curz > maxz) {
					maxz = curz;
				}
			}
			miny = -miny;
			m_rbInitTransVec = new BulletSharp.Math.Vector3(0, miny + bodyMargin + m_groundMargin, 0);
			m_rbInitTrans = Matrix.Translation(m_rbInitTransVec); //* Matrix.RotationY(Random.Range(0.0f, 360.0f));

			//float gtInertiaX = (1.0f / 12.0f) * randomMass * (3.0f * maxz * maxz + (2.0f * miny) * (2.0f * miny));
			//float gtInertiaZ = gtInertiaX;
			//float gtInertiaY = 0.5f * randomMass * maxz * maxz;
			//BulletSharp.Math.Vector3 gtInertia = new BulletSharp.Math.Vector3(gtInertiaX, gtInertiaY, gtInertiaZ);
			//Debug.Log("GT INERTIA: " + gtInertia.ToString());
			//randomInertia = gtInertia;

			m_rb = CreateRigidBody(randomMass, randomInertia, m_rbInitTrans, m_cs, bodyMat, bodyFriction, viz: RENDER_MODE);
			//m_rb = CreateRigidBody(randomMass, m_rbInitTrans, m_cs, bodyMat, bodyFriction);
			m_rb.AngularFactor = angularFactor;
			m_rb.SetSleepingThresholds(linearSleepThresh, angularSleepThresh);

			m_mass = randomMass;
			m_inertia = randomInertia;
			m_density = randomDensity;

		} else {
			// using the same mesh just need to choose a new density
			// steps for determinism
			// https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=3143
			

			float randomDensity;
			if (useConstantDensity) {
				randomDensity = densityMin;
			} else {
				randomDensity = Random.Range(densityMin, densityMax);
			}
            randomMass = randomDensity * m_masses[m_curObjIdx];
			BulletSharp.Math.Vector3 objInertiaInfo = m_inertias[m_curObjIdx];
            float inertiax = randomDensity * (objInertiaInfo[1] + objInertiaInfo[2]);
            float inertiay = randomDensity * (objInertiaInfo[0] + objInertiaInfo[2]);
            float inertiaz = randomDensity * (objInertiaInfo[0] + objInertiaInfo[1]);
            randomInertia = new BulletSharp.Math.Vector3(inertiax, inertiay, inertiaz);

			m_rb.SetMassProps(randomMass, randomInertia);

			m_rbInitTrans = Matrix.Translation(m_rbInitTransVec);// * Matrix.RotationY(Random.Range(0.0f, 360.0f));
			m_rb = ResetRigidBody(m_rb, randomMass, randomInertia, m_rbInitTrans, m_cs, bodyFriction);
			m_rb.AngularFactor = angularFactor;
			m_rb.SetSleepingThresholds(linearSleepThresh, angularSleepThresh);

			// HingeConstraint hingeConstraint = new HingeConstraint(m_rb, new BulletSharp.Math.Vector3(0.0f), new BulletSharp.Math.Vector3(0.0f, 1.0f, 0.0f), false);
            // m_world.AddConstraint(hingeConstraint);

			// DestroySimObj(); // have to do this to set mass properties but can reuse previously calculated everything else
			// if (m_cs == null) Debug.Log("NOT NULL");
			// m_rb = CreateRigidBody(randomMass, randomInertia, m_rbInitTrans, m_cs, bodyMat, bodyFriction);
			// m_rb.AngularFactor = new BulletSharp.Math.Vector3(angularFactor);
			// m_rb.SetSleepingThresholds(linearSleepThresh, angularSleepThresh);

			m_mass = randomMass;
			m_inertia = randomInertia;
			m_density = randomDensity;

		}

		m_stepCount = 0;

		m_broadphase.ResetPool(m_colDispatcher);
		m_solver.Reset();

		float curMass = 1.0f / m_rb.InvMass;
		if (DEBUG) Debug.Log("Mass: " + curMass.ToString());
		if (DEBUG) Debug.Log("LOCAL MOMENT: " + m_rb.LocalInertia.ToString());
		if (DEBUG) Debug.Log("COM " + m_rb.CenterOfMassPosition.ToString());
		if (DEBUG) Debug.Log("Density " + m_density.ToString());

		// determine impulse position
		ClosestRayResultCallback cb;
		BulletSharp.Math.Vector3 vertexNormal = new BulletSharp.Math.Vector3();
		int missCount = 0;
		do
		{
			// choose random vertex to apply force to
			// pick a random point around in the plane around y position
			float offsetx = UnityEngine.Random.Range(-100.0f, 100.0f);
			float offsetz = UnityEngine.Random.Range(-100.0f, 100.0f);
			Vector2 offsetvec = new Vector2(offsetx, offsetz);
			// offsetvec.Normalize();
			//float relForceHeight = 0.75f;
			UnityEngine.Vector3 offsetPt = new UnityEngine.Vector3(offsetvec[0],
															m_rb.CenterOfMassPosition.Y,
															offsetvec[1]);

			BulletSharp.Math.Vector3 btOffsetPt = BSExtensionMethods2.ToBullet(offsetPt);
			BulletSharp.Math.Vector3 btInnerPt = m_rb.CenterOfMassPosition;
			cb = new ClosestRayResultCallback(ref btOffsetPt, ref btInnerPt);

			// Debug.DrawLine(BSExtensionMethods2.ToUnity(btInnerPt), offsetPt, Color.red, 2.0f);

			m_world.RayTest(btOffsetPt, btInnerPt, cb);
			if (cb.HasHit)
			{
				m_forcePoint = cb.HitPointWorld;
				vertexNormal = cb.HitNormalWorld;
			}
			else
			{
				missCount++;
				//Debug.Log("ERROR - couldn't find point to apply force to. Retrying...");
				//return;
			}
		} while (!cb.HasHit);

		if (DEBUG) Debug.Log("Missed impulse " + missCount.ToString() + " times.");
		if (DEBUG) Debug.LogFormat("ForcePoint: " + m_forcePoint.ToString());

		// get force vector
		// loop until force is applied to outside of object
		UnityEngine.Vector3 uForceVec = new UnityEngine.Vector3();

		// initialize force vector to coincide with center of mass
		BulletSharp.Math.Vector3 btForceVec = m_rb.CenterOfMassPosition - m_forcePoint;
        // then randomly vary it within the x/z plane to be within the specified distance
		BulletSharp.Math.Vector3 btVariationVec = new BulletSharp.Math.Vector3(-btForceVec[2], 0.0f, btForceVec[0]);
		btVariationVec.Normalize();
		float varyForce;
		BulletSharp.Math.Vector3 proposedForceVec;
		do {
			varyForce = UnityEngine.Random.Range(-forceDistMax, forceDistMax);
			proposedForceVec = btVariationVec * varyForce + btForceVec;
		} while (proposedForceVec.Dot(vertexNormal) >= 0); // must also be on the outside of the object
		btForceVec = proposedForceVec;
		btForceVec.Normalize();
		uForceVec = BSExtensionMethods2.ToUnity(btForceVec);
		if (DEBUG) Debug.Log("FORCE DIST: " + varyForce.ToString());

		//UnityEngine.Vector3 uVtxNormal = BSExtensionMethods2.ToUnity(vertexNormal);
		//uVtxNormal.Normalize();
		//do
		//{
		//	float forcex = UnityEngine.Random.Range(-1.0f, 1.0f);
		//	float forcez = UnityEngine.Random.Range(-1.0f, 1.0f);
		//	uForceVec.Set(forcex, 0.0f, forcez);
		//	uForceVec.Normalize();
		//} while (UnityEngine.Vector3.Dot(uForceVec, uVtxNormal) >= 0);
		// random constrained magnitude
		float mag = UnityEngine.Random.Range(impulseMin, impulseMax);
		//Debug.Log("Vol: " + objectVolume.ToString());
		// if (varyScale) {
		// 	mag *= randomMass; // scale impulse t unity
		//according to object scale
		// } else {
		// 	mag *= curMass;
		// }
		mag *= m_mass; // scale impulse according to object mass
		uForceVec *= mag;

		// set directly for debugging
		 //uForceVec.Set(2.5f, 0.0f, 0.0f);
		 //m_forcePoint = new BulletSharp.Math.Vector3(0.0f, m_rb.CenterOfMassPosition.Y, -0.15f);

		m_forceVec = BSExtensionMethods2.ToBullet(uForceVec);

		if (DEBUG) Debug.LogFormat("ForceVec: " + m_forceVec.ToString());

		if (DEBUG)
		{
			UnityEngine.Vector3 debugVec = -uForceVec;
			debugVec.Scale(new UnityEngine.Vector3(0.5f, 0.5f, 0.5f));
			Debug.DrawRay(BSExtensionMethods2.ToUnity(m_forcePoint), debugVec, Color.green, 1.0f);
			Debug.DrawLine(BSExtensionMethods2.ToUnity(m_rb.CenterOfMassPosition), BSExtensionMethods2.ToUnity(m_forcePoint), Color.cyan, 1.0f);
			Debug.DrawLine(BSExtensionMethods2.ToUnity(m_rb.CenterOfMassPosition), BSExtensionMethods2.ToUnity(m_rb.CenterOfMassPosition) + BSExtensionMethods2.ToUnity(btVariationVec) * varyForce, Color.blue, 1.0f);
		}

		// apply the random impulse
		BulletSharp.Math.Vector3 radius = m_forcePoint - m_rb.CenterOfMassPosition;
		m_rb.ApplyImpulse(m_forceVec, radius);
		// m_rb.ApplyTorqueImpulse(new BulletSharp.Math.Vector3(0.0f, 1.0f, 0.0f));
		// m_rb.ApplyCentralImpulse(new BulletSharp.Math.Vector3(4.0f, 0.0f, 2.0f));
		// BulletSharp.Math.Vector3 newAngVel = m_rb.AngularVelocity;
		// newAngVel.X = 0.0f;
		// newAngVel.Z = 0.0f;
		// m_rb.AngularVelocity = newAngVel;

		// calculate ground truth for debugging
		//BulletSharp.Math.Vector3 gtAngVel = radius.Cross(m_forceVec) / m_inertia;
		//BulletSharp.Math.Vector3 gtLinVel = m_forceVec / m_mass;
		//Debug.Log("GT LIN VEL: " + gtLinVel.ToString());
		//Debug.Log("GT ANG VEL: " + gtAngVel.ToString());
		     
	}

	protected void InitStateLists() {
		m_stepVel = new List<UnityEngine.Vector3>();
		m_stepAngVel = new List<UnityEngine.Vector3>();
		m_stepRot = new List<UnityEngine.Quaternion>();
		m_stepEulerRot = new List<UnityEngine.Vector3>();
		m_stepPos = new List<UnityEngine.Vector3>();
		m_stepNum = new List<int>();
	}

	protected void ClearStateLists() {
		m_stepVel.Clear();
		m_stepAngVel.Clear();
		m_stepRot.Clear();
		m_stepEulerRot.Clear();
		m_stepPos.Clear();
		m_stepNum.Clear();
	}

	// The save state functions save various information about the rigid
	// body simulation to global variables to be written out.
	protected void SaveInitState() {
		m_com0 = m_rb.CenterOfMassPosition;
		Matrix curTrans = m_rb.WorldTransform;
		m_pos0 = BSExtensionMethods2.ExtractTranslationFromMatrix(ref curTrans);
		m_vel0 = m_rb.LinearVelocity;
		m_angvel0 = m_rb.AngularVelocity;
		if (DEBUG) Debug.Log("Init ANGVEL: " + m_angvel0.ToString());
		if (DEBUG) Debug.Log("Init ANGVEL len: " + (m_angvel0.Length*Time.fixedDeltaTime).ToString());
		m_rot0 = BSExtensionMethods2.ExtractRotationFromMatrix(ref curTrans);
		m_eulerrot0 = m_rot0.eulerAngles;
	}

	protected void SaveFinalState() {
		m_comf = m_rb.CenterOfMassPosition;
		Matrix curTrans = m_rb.WorldTransform;
		m_posf = BSExtensionMethods2.ExtractTranslationFromMatrix(ref curTrans);
		m_rotf = BSExtensionMethods2.ExtractRotationFromMatrix(ref curTrans);
		m_eulerrotf = m_rotf.eulerAngles;
	}

	protected void SaveCurrentState() {
		m_stepVel.Add(BSExtensionMethods2.ToUnity(m_rb.LinearVelocity));
		m_stepAngVel.Add(BSExtensionMethods2.ToUnity(m_rb.AngularVelocity));
		Matrix curTrans = m_rb.WorldTransform;
		m_stepRot.Add(BSExtensionMethods2.ExtractRotationFromMatrix(ref curTrans));
		m_stepEulerRot.Add(BSExtensionMethods2.ToUnity(m_totalRot));
		m_stepPos.Add(BSExtensionMethods2.ToUnity(m_rb.CenterOfMassPosition));
		m_stepNum.Add(m_stepCount);
	}

	protected void DumpSimInfo() {
		// write out to json
		if (DEBUG) Debug.Log("Finished Trial " + (m_curSimNum + 1) + ", Saving Data...");
		// always print out status every 250 sims
		if (m_curSimNum % 50 == 0)
		{
			Debug.Log("Finished Sim " + m_curSimNum + " for object " + m_curObjIdx + "...");
		}
		// output to data file
		SimSaveData data = new SimSaveData();
		data.shape = dataInDir + "/" + m_objFiles[m_curObjIdx].Replace(".obj","");
		data.scale = m_pclScale;
		data.mass = m_mass;
		data.density = m_density;
		data.inertia = BSExtensionMethods2.ToUnity(m_inertia);
		data.forcePoint = BSExtensionMethods2.ToUnity(m_forcePoint);
		data.forceVec = BSExtensionMethods2.ToUnity(m_forceVec);
		data.vel0 = BSExtensionMethods2.ToUnity(m_vel0);
		data.angvel0 = BSExtensionMethods2.ToUnity(m_angvel0);
		data.com0 = BSExtensionMethods2.ToUnity(m_com0);
		data.pos0 = m_pos0;
		data.rot0 = m_rot0;
		data.eulerrot0 = m_eulerrot0;
		data.comf = BSExtensionMethods2.ToUnity(m_comf);
		data.posf = m_posf;
		data.rotf = m_rotf;
		data.eulerrotf = m_eulerrotf;
		data.totalRot = BSExtensionMethods2.ToUnity(m_totalRot);
		data.stepVel = m_stepVel;
		data.stepAngVel = m_stepAngVel;
		data.stepRot = m_stepRot;
		data.stepEulerRot = m_stepEulerRot;
		data.stepPos = m_stepPos;
		data.stepNum = m_stepNum;

		string dataStr = data.SaveString();
		if (DEBUG) Debug.Log(dataStr);

		string outFile = m_curDataOutDir + "sim_" + m_curSimNum.ToString() + ".json";
		StreamWriter writer = new StreamWriter(outFile, false);
		writer.WriteLine(dataStr);
		writer.Close();
	}

	protected void DumpFallenObjects() {
		StringListJsonData data = new StringListJsonData();
		data.stringList = m_fallenFiles;
		string dataStr = data.SaveString();

		string outFile = Path.Combine(dataOutDir, "fallen_objects.json");
		Debug.Log("writing fallen objects to " + outFile);
		StreamWriter writer = new StreamWriter(outFile, false);
		writer.WriteLine(dataStr);
		writer.Close();
	}

	// Object to store simulation data to be written out
	private class SimSaveData
	{
		public string shape; // the shape used in this sim
		public UnityEngine.Vector3 scale; // scale of the object in all dimensions
		public float density; // the density of the object material
		public float mass; // the mass used to simulate the object
		public UnityEngine.Vector3 inertia; // the object moment of inertia around the principal axes
		public UnityEngine.Vector3 com0; // COM pos (x, y, z) at first frame
		public UnityEngine.Vector3 pos0; // COM pos (x, y, z) at first frame
		public UnityEngine.Quaternion rot0; // object rotation (w, x, y, z) at first frame
		public UnityEngine.Vector3 eulerrot0; // euler rotation at first frame
		public UnityEngine.Vector3 comf; // last frame
		public UnityEngine.Vector3 posf; // last frame
		public UnityEngine.Quaternion rotf; // last frame
		public UnityEngine.Vector3 eulerrotf; // euler rotation at final frame
		public UnityEngine.Vector3 totalRot; // total euler rotation over entire sim
		public UnityEngine.Vector3 forcePoint; // (x, y, z) point where the force was applied
		public UnityEngine.Vector3 forceVec; // force vector (x, y, z)
		public UnityEngine.Vector3 vel0; // initial velocity
		public UnityEngine.Vector3 angvel0; // initial angular velocity
		public List<UnityEngine.Vector3> stepVel;
		public List<UnityEngine.Vector3> stepAngVel;
		public List<UnityEngine.Quaternion> stepRot;
		public List<UnityEngine.Vector3> stepEulerRot;
		public List<UnityEngine.Vector3> stepPos;
		public List<int> stepNum;

		public string SaveString()
		{
			return JsonUtility.ToJson(this);
		}

		public static SimSaveData CreateFromJSON(string jsonString) {
			return JsonUtility.FromJson<SimSaveData>(jsonString);
		}
	}

	// Object for reading from obj info json
	private class ObjInfo {
		public float vol;
		public int num_vox;
		public float[] inertia;

		public static ObjInfo CreateFromJSON(string jsonString) {
        	return JsonUtility.FromJson<ObjInfo>(jsonString);
		}
	}

	private class StringListJsonData {

		public List<string> stringList;

		public string SaveString()
		{
			return JsonUtility.ToJson(this);
		}
	}
}
