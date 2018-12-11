using BulletSharp.Math;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

public class DataGenUtils
{
	public struct BulletOBJMesh
	{
		public float[] vertices;
		public int[] indices;

		public BulletOBJMesh Scale(float x, float y, float z) {
			BulletOBJMesh scaledMesh = new BulletOBJMesh();
			scaledMesh.vertices = new float[vertices.Length];
			scaledMesh.indices = new int[indices.Length];
			vertices.CopyTo(scaledMesh.vertices, 0);
			indices.CopyTo(scaledMesh.indices, 0);

			for (int i = 0; i < (int)scaledMesh.vertices.Length / 3; i++) {
				scaledMesh.vertices[i*3 + 0] *= x;
				scaledMesh.vertices[i*3 + 1] *= y;
				scaledMesh.vertices[i*3 + 2] *= z;
			}

			return scaledMesh;
		}
	}

	// Converts an OBJMesh (assumed to be a tri-mesh) to a bullet-friendly structure
	public static BulletOBJMesh BulletMeshFromUnity(OBJLoader.OBJMesh mesh) {
		BulletOBJMesh btmesh = new BulletOBJMesh();

		btmesh.vertices = new float[mesh.vertices.Count * 3];
		for (int i = 0; i < mesh.vertices.Count; i++) {
			UnityEngine.Vector3 curVert = mesh.vertices[i];
			btmesh.vertices[i*3 + 0] = curVert[0];
			btmesh.vertices[i*3 + 1] = curVert[1];
			btmesh.vertices[i*3 + 2] = curVert[2];
		}
			
		List<int> indList = new List<int>();
		for (int i = 0; i < mesh.faces.Count; i++) {
			int[] faceIndices = mesh.faces[i].indexes;
			for (int j = 0; j < faceIndices.Length; j++) {
				int idx = faceIndices[j];
				indList.Add(faceIndices[j]);
			}
		}

		btmesh.indices = indList.ToArray();
		
		return btmesh;
	}

	// Converts an OBJMesh (assumed to be a tri-mesh) to a bullet-friendly structure
    public static Mesh ToUnityMesh(OBJLoader.OBJMesh objMesh)
    {
        Mesh mesh = new Mesh();
        UnityEngine.Vector3[] newVertices = new UnityEngine.Vector3[objMesh.faces.Count * 3]; // must make a separate copy of each for normals to be correct
		//UnityEngine.Vector2[] newUV = new UnityEngine.Vector2[objMesh.faces.Count * 3];
		int[] newTriangles = new int[objMesh.faces.Count * 3];

        for (int i = 0; i < objMesh.vertices.Count; i++) {
            newVertices[i] = objMesh.vertices[i];
        }
        for (int i = 0; i < objMesh.faces.Count; i++) {
			UnityEngine.Vector3 v1 = objMesh.vertices[objMesh.faces[i].indexes[0]];
			UnityEngine.Vector3 v2 = objMesh.vertices[objMesh.faces[i].indexes[1]];
			UnityEngine.Vector3 v3 = objMesh.vertices[objMesh.faces[i].indexes[2]];

			newVertices[i*3 + 0] = v1;
			newVertices[i*3 + 1] = v2;
			newVertices[i*3 + 2] = v3;

            newTriangles[i*3 + 0] = i*3 + 0;
            newTriangles[i*3 + 1] = i*3 + 1;
            newTriangles[i*3 + 2] = i*3 + 2;

			//UnityEngine.Vector3 normal = UnityEngine.Vector3.Cross(v3 - v1, v2 - v1);
			//UnityEngine.Quaternion rot = UnityEngine.Quaternion.Inverse(UnityEngine.Quaternion.LookRotation(normal));

   //         // Assign the uvs, applying a scale factor to control the texture tiling.
			//float scaleFactor = 2.0f;
			//newUV[i*3 + 0]     = (Vector2)(rot * v1) * scaleFactor;
			//newUV[i*3 + 1] = (Vector2)(rot * v2) * scaleFactor;
			//newUV[i*3 + 2] = (Vector2)(rot * v3) * scaleFactor;
        }

        mesh.vertices = newVertices;
		//mesh.uv = newUV;
        mesh.triangles = newTriangles;
        mesh.RecalculateBounds();
        mesh.RecalculateNormals();

		//mesh.uv = UnityEditor.Unwrapping.GeneratePerTriangleUV(mesh);

        return mesh;
    }
}
