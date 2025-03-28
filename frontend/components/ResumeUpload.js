import { useState } from "react";
import axios from "axios";

const ResumeUpload = () => {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState("");

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) {
            setMessage("Please select a resume file to upload.");
            return;
        }

        const formData = new FormData();
        formData.append("resume", file);

        try {
            const response = await axios.post("http://127.0.0.1:8000/api/upload-resume/", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setMessage(response.data.message);
        } catch (error) {
            setMessage("Error uploading file. Please try again.");
            console.error("Upload Error:", error);
        }
    };

    return (
        <div style={{ padding: "20px", textAlign: "center" }}>
            <h2>Upload Resume</h2>
            <input type="file" onChange={handleFileChange} accept=".pdf,.doc,.docx" />
            <button onClick={handleUpload} style={{ marginLeft: "10px", padding: "5px 10px", cursor: "pointer" }}>
                Upload
            </button>
            {message && <p>{message}</p>}
        </div>
    );
};

export default ResumeUpload;
