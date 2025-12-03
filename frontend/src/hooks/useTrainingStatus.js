import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const useTrainingStatus = () => {
    const [isTraining, setIsTraining] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusMessage, setStatusMessage] = useState('');
    const [isTrained, setIsTrained] = useState(false);
    const [error, setError] = useState(null);

    // Use a ref to keep track of the interval so we can clear it if needed
    const intervalRef = useRef(null);

    const fetchStatus = async () => {
        try {
            const response = await axios.get('http://localhost:8000/status');
            const { is_training, progress, message, is_trained } = response.data;

            setIsTraining(is_training);
            setProgress(progress);
            setStatusMessage(message);
            setIsTrained(is_trained);
            setError(null);
        } catch (err) {
            console.error("Failed to fetch training status:", err);
            setError("Failed to connect to server.");
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

    return { isTraining, progress, statusMessage, isTrained, error };
};

export default useTrainingStatus;
