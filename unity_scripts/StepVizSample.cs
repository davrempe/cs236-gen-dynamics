using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class StepVizSample : MonoBehaviour {

	public bool BUILD_MODE = false;

	// directory in Resources where .json files of prediction results are
    public string dataInDir = "PredictionVizData";
	public int startSimIdx = 0;
	public int endSimIdx = -1;
	// directory to store images of results
	public string dataOutDir = "./data_out";

	// object appearance
	public Material groundMat;
	public Material initMat;
    public Material gtMat;
    public Material sampMat;

    // show the error
	public Text errorText;
    
	public Vector3 perspCameraLoc = new Vector3(1, 1, -4); 
    public Vector3 perspCameraTarget = new Vector3(0, 0, 0);

    // base directory for all possible meshes we could need
	protected string m_baseModelDir;
	protected string m_imgOutDir;
	protected string m_simOutdir;
    // the sim we're on
	protected int m_simNum;
	protected int m_frameNum;
	protected int m_totalFrames;
    
	protected Mesh m_mesh;
	protected GameObject m_initObj;
	protected GameObject m_gtObj;
	protected GameObject m_sampObj;
    
	private PredInfo m_curPred;

	protected bool m_isQuit;
	protected bool firstFrame;

	// Use this for initialization
	void Start () {
		m_isQuit = false;
		// create data directory if doesn't exist
		if (!BUILD_MODE) {
		    m_imgOutDir = Path.Combine(Application.dataPath, "../");
			m_imgOutDir = Path.Combine(m_imgOutDir, dataOutDir);
		} else {
			m_imgOutDir = Path.Combine(Application.dataPath, dataOutDir);
		}
		Directory.CreateDirectory(m_imgOutDir);

        // location of shape objects
        if (!BUILD_MODE) {
			m_baseModelDir = Application.dataPath + "/Resources/";
        } else {
			m_baseModelDir = Application.dataPath + "/../../Assets/Resources/";
        }

		SetEyeTarget(perspCameraLoc, perspCameraTarget);
        
		m_simNum = startSimIdx;

        // create ground
		GameObject groundObj = GameObject.CreatePrimitive(PrimitiveType.Plane);
		groundObj.transform.localScale = new Vector3(20.0f, 1.0f, 20.0f);
		groundObj.transform.Rotate(new Vector3(0.0f, 1.0f, 0.0f), 0.0f);
		groundObj.GetComponent<MeshRenderer>().sharedMaterial = groundMat;

		firstFrame = true;

        // set up first simulation
		//m_simNum = 0;
		m_frameNum = -1;
		m_simOutdir = Path.Combine(m_imgOutDir, "sim_" + m_simNum.ToString());
		Directory.CreateDirectory(m_simOutdir);
		LoadSim(m_simNum);
	}

	void LoadSim(int idx) {
		// read in results from json
		string predFile = Path.Combine(dataInDir, "eval_sim_" + idx.ToString());
        Debug.Log("Visualizing " + predFile);
        var jsonTextFile = Resources.Load<TextAsset>(predFile);
		m_curPred = PredInfo.CreateFromJSON(jsonTextFile.ToString());

        // read in mesh to use
		string meshFile = Path.Combine(m_baseModelDir, m_curPred.shape);
        meshFile += ".obj";
        Debug.Log(meshFile);

        OBJLoader.OBJMesh objLoaderMesh = OBJLoader.LoadOBJMesh(meshFile);
        Debug.Assert(objLoaderMesh.vertices.Count > 0);
        Debug.Assert(objLoaderMesh.faces.Count > 0);
        m_mesh = DataGenUtils.ToUnityMesh(objLoaderMesh);

		// create init object
		m_initObj = new GameObject("Sim" + idx.ToString() + "_Init");
		MeshFilter mfInit = m_initObj.AddComponent<MeshFilter>();
		mfInit.mesh = m_mesh;
		MeshRenderer mrInit = m_initObj.AddComponent<MeshRenderer>();
		mrInit.sharedMaterial = initMat;
		m_initObj.transform.localScale = m_curPred.scale;
		m_initObj.transform.position = m_curPred.gt_pos[0];
		m_initObj.transform.eulerAngles = m_curPred.gt_rot[0];

        // create GT object
		m_gtObj = new GameObject("Sim" + idx.ToString() + "_GT");
		MeshFilter mf = m_gtObj.AddComponent<MeshFilter>();
        mf.mesh = m_mesh;
        MeshRenderer mr = m_gtObj.AddComponent<MeshRenderer>();
		mr.sharedMaterial = gtMat;
		m_gtObj.transform.localScale = m_curPred.scale;
		m_gtObj.transform.position = m_curPred.gt_pos[0];
		m_gtObj.transform.eulerAngles = m_curPred.gt_rot[0];

        // create sampled object
		m_sampObj = new GameObject("Sim" + idx.ToString() + "_Samp");
		MeshFilter mf2 = m_sampObj.AddComponent<MeshFilter>();
		mf2.mesh = m_mesh;
		MeshRenderer mr2 = m_sampObj.AddComponent<MeshRenderer>();
		mr2.sharedMaterial = sampMat;
		m_sampObj.transform.localScale = m_curPred.scale;
		m_sampObj.transform.position = m_curPred.samp_pos[0];
		m_sampObj.transform.eulerAngles = m_curPred.samp_rot[0];
	}
	
	// Update is called once per frame
	void Update () {
		if (firstFrame) {
			// can't saved screenshots on first frame for some reason
			firstFrame = false;
			return;
		}

        // set up next frame
		m_frameNum++;
		// check if we're done with the current simulation
        if (m_frameNum >= m_curPred.samp_pos.Count) {
			Destroy(m_initObj);
            Destroy(m_gtObj);
            Destroy(m_sampObj);
            Destroy(m_mesh);
            // set up next sim
			if (endSimIdx != -1 && m_simNum+1 == endSimIdx) {
                m_isQuit = true;
                Application.Quit();
            } else {
                m_simNum++;
            }
            m_frameNum = 0;
            m_simOutdir = Path.Combine(m_imgOutDir, "sim_" + m_simNum.ToString());
			Directory.CreateDirectory(m_simOutdir);
            LoadSim(m_simNum);
		} else {
            // set up next frame
			m_gtObj.transform.position = m_curPred.gt_pos[m_frameNum];
			m_gtObj.transform.eulerAngles = m_curPred.gt_rot[m_frameNum];
			m_sampObj.transform.position = m_curPred.samp_pos[m_frameNum];
			m_sampObj.transform.eulerAngles = m_curPred.samp_rot[m_frameNum];

			// display error
			if (errorText != null) {
    			errorText.text = "PosErr: " + m_curPred.pos_err.ToString() + "\n RotErr: " + m_curPred.rot_err.ToString();
                Debug.Log(errorText.text);
			}
		}

		// take a picture of the setup
        string outFile = Path.Combine(m_simOutdir, "frame_" + m_frameNum.ToString().PadLeft(6, '0') + ".png");
        ScreenCapture.CaptureScreenshot(outFile, 1);
        Debug.Log(outFile);
	}

	protected void SetEyeTarget(Vector3 eye, Vector3 targ) {
        Transform t = Camera.main.transform;
        t.position = eye;
        t.rotation = Quaternion.LookRotation((targ - eye).normalized, Vector3.up);
    }

	// Object for reading from prediction info json
    private class PredInfo
    {
		public string shape;
		public Vector3 scale;
		public List<Vector3> gt_pos;
        public List<Vector3> samp_pos;
		public List<Vector3> gt_rot;
        public List<Vector3> samp_rot;
		public float pos_err;
		public float rot_err;

		public static PredInfo CreateFromJSON(string jsonString)
        {
			return JsonUtility.FromJson<PredInfo>(jsonString);
        }
    }
}
