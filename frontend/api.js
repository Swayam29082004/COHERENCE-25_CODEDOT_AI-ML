const API_URL = "http://127.0.0.1:8000";  // Ensure this is correct

export const uploadResume = async (formData) => {
  try {
    const response = await fetch(`${API_URL}/upload-resume/`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Failed to upload resume");
    }

    return await response.json();
  } catch (error) {
    console.error("Upload error:", error);
    throw error;
  }
};


