import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const useTrainingStatus = () => {
    const [isTraining, setIsTraining] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState('');
    const [isTrained, setIsTrained] = useState(false);
    const [error, setError] = useState(null);

    const [logs, setLogs] = useState([]);

    // Use a ref to keep track of the interval so we can clear it if needed
    const intervalRef = useRef(null);

    const fetchStatus = async () => {
        try {
            const response = await axios.get('http://localhost:8000/status');
            const data = response.data;

            // New TrainingManager state structure
            const statusStr = data.status || 'idle';
            const isTrain = statusStr === 'training';

            setIsTraining(isTrain);
            setProgress(data.progress);
            setStatusMessage(data.message);
            setIsTrained(data.is_trained);
            setLogs(data.logs || []);

            if (statusStr === 'failed' && data.error) {
                setError(data.error); // { message, suggestion, context }
            } else if (statusStr === 'failed') {
                setError({ message: data.message, suggestion: "An unknown error occurred." });
            } else {
                setError(null);
            }

        } catch (err) {
            console.error("Failed to fetch training status:", err);
            // Don't overwrite error if we just have a transient network glitch, unless it persists?
            // For now, let's just log it.
        }
    };

    useEffect(() => {
        // Initial fetch
        fetchStatus();

        // Poll every 2 seconds
        intervalRef.current = setInterval(fetchStatus, 2000);

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, []);

    return { isTraining, progress, statusMessage, isTrained, error, logs };
};

export default useTrainingStatus;
