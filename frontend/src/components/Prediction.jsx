import React, { useState, useEffect } from 'react';
import {
    Box, TextField, Button, Typography, Paper, Grid,
    Tabs, Tab, Slider, IconButton, Stack, Alert, Switch, Chip
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import axios from 'axios';
import { motion } from 'framer-motion';

const ALL_ELEMENTS = ['Ag', 'Al', 'B', 'Ba', 'Bi', 'C', 'Ca', 'Fe', 'Hf', 'Ho', 'K',
    'Li', 'Mn', 'Na', 'Nb', 'O', 'Pr', 'Sb', 'Sc', 'Sr', 'Ta', 'Ti',
    'Zn', 'Zr'];

const Prediction = ({ isTraining, isTrained }) => {
    const [tabValue, setTabValue] = useState(0);
    const [formula, setFormula] = useState('');
    const [builderElements, setBuilderElements] = useState([]);
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);
    const [showComposition, setShowComposition] = useState(false);

    const [activeModel, setActiveModel] = useState(null);

    useEffect(() => {
        const fetchActiveModel = async () => {
            try {
                const response = await axios.get('http://localhost:8000/active-model');
                setActiveModel(response.data);
            } catch (err) {
                console.error("Failed to fetch active model info:", err);
            }
        };
        fetchActiveModel();
    }, [isTrained]);

    const isDisabled = isTraining || !isTrained;

    const handleTabChange = (event, newValue) => {
        setTabValue(newValue);
        setPrediction(null);
        setError(null);
    };

    const handlePredict = async () => {
        setError(null);
        let formulaToPredict = formula;

        if (tabValue === 1) {
            // Construct formula from builder
            if (builderElements.length === 0) {
                setError("Please add at least one element.");
                return;
            }
            formulaToPredict = builderElements.map(e => `${e.element}${e.amount}`).join('');
        }

        if (!formulaToPredict) {
            setError("Please enter a formula.");
            return;
        }

        try {
            const response = await axios.post('http://localhost:8000/predict', { formula: formulaToPredict });
            setPrediction(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || "Prediction failed. Ensure models are trained.");
        }
    };

    const addElement = (element) => {
        setBuilderElements([...builderElements, { element, amount: 1 }]);
    };

    const removeElement = (index) => {
        const newElements = [...builderElements];
        newElements.splice(index, 1);
        setBuilderElements(newElements);
    };

    const updateAmount = (index, newAmount) => {
        const newElements = [...builderElements];
        newElements[index].amount = newAmount;
        setBuilderElements(newElements);
    };

    const resetBuilder = () => {
        setBuilderElements([]);
        setPrediction(null);
        setError(null);
    };

    return (
        <Box sx={{ width: '100%', maxWidth: 1000, mx: 'auto' }}>
            <Box sx={{ textAlign: 'center', mb: 6 }}>
                <Typography variant="h3" gutterBottom color="primary">Property Prediction</Typography>
                <Typography variant="body1" color="text.secondary">
                    Enter a chemical formula or build one interactively to predict its properties.
                </Typography>
            </Box>

            {activeModel && (
                <Paper sx={{ p: 2, mb: 4, bgcolor: '#e8f5e9', border: '1px solid #c8e6c9', borderRadius: 2 }}>
                    <Typography variant="subtitle1" fontWeight="bold" color="success.dark" gutterBottom>
                        Active Models
                    </Typography>
                    <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                            <Typography variant="body2">
                                <strong>d<sub>33</sub>:</strong> {activeModel.d33.name} ({activeModel.d33.mode})
                            </Typography>
                        </Grid>
                        <Grid item xs={12} sm={6}>
                            <Typography variant="body2">
                                <strong>T<sub>c</sub>:</strong> {activeModel.Tc.name} ({activeModel.Tc.mode})
                            </Typography>
                        </Grid>
                    </Grid>
                </Paper>
            )}

            <Paper sx={{ p: 4, mb: 4, borderRadius: 4, position: 'relative' }}>
                {isDisabled && (
                    <Box sx={{
                        position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                        bgcolor: 'rgba(255,255,255,0.7)', zIndex: 10,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        borderRadius: 4, flexDirection: 'column', gap: 2
                    }}>
                        <Alert severity="warning" variant="filled" sx={{ width: '80%' }}>
                            {isTraining
                                ? "Model is currently retraining. Prediction is disabled until training completes."
                                : "Model is not trained. Please go to the Retraining section to train the model first."}
                        </Alert>
                    </Box>
                )}

                <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    centered
                    sx={{ mb: 4, '& .MuiTab-root': { fontSize: '1.1rem' } }}
                >
                    <Tab label="Direct Input" disabled={isDisabled} />
                    <Tab label="Interactive Builder" disabled={isDisabled} />
                </Tabs>

                {tabValue === 0 && (
                    <Box sx={{ maxWidth: 600, mx: 'auto' }}>
                        <TextField
                            fullWidth
                            label="Chemical Formula"
                            variant="outlined"
                            value={formula}
                            onChange={(e) => setFormula(e.target.value)}
                            placeholder="e.g., 0.96(K0.5Na0.5)NbO3-0.04Bi0.5Na0.5TiO3"
                            sx={{ mb: 2 }}
                            disabled={isDisabled}
                        />
                    </Box>
                )}

                {tabValue === 1 && (
                    <Box>
                        <Typography variant="subtitle1" gutterBottom fontWeight="bold">Select Elements:</Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 4 }}>
                            {ALL_ELEMENTS.map((el) => (
                                <Button
                                    key={el}
                                    variant="outlined"
                                    size="small"
                                    onClick={() => addElement(el)}
                                    sx={{ borderRadius: 2 }}
                                    disabled={isDisabled}
                                >
                                    {el}
                                </Button>
                            ))}
                        </Box>

                        {builderElements.length > 0 && (
                            <>
                                <Typography variant="subtitle1" gutterBottom fontWeight="bold">Composition:</Typography>
                                <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                                    {builderElements.map((item, index) => (
                                        <Grid container spacing={2} alignItems="center" key={index} sx={{ mb: 1 }}>
                                            <Grid size={{ xs: 2 }}>
                                                <Typography fontWeight="bold">{item.element}</Typography>
                                            </Grid>
                                            <Grid size={{ xs: 8 }}>
                                                <Slider
                                                    value={item.amount}
                                                    min={0}
                                                    max={5}
                                                    step={0.01}
                                                    valueLabelDisplay="auto"
                                                    onChange={(e, val) => updateAmount(index, val)}
                                                    disabled={isDisabled}
                                                />
                                            </Grid>
                                            <Grid size={{ xs: 2 }}>
                                                <IconButton onClick={() => removeElement(index)} color="error" disabled={isDisabled}>
                                                    <DeleteIcon />
                                                </IconButton>
                                            </Grid>
                                        </Grid>
                                    ))}
                                </Paper>
                                <Button startIcon={<RestartAltIcon />} onClick={resetBuilder} color="warning" disabled={isDisabled}>
                                    Reset Builder
                                </Button>
                            </>
                        )}
                    </Box>
                )}

                <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center', flexDirection: 'column', alignItems: 'center' }}>
                    {tabValue === 0 && (
                        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" sx={{ mr: 1 }}>Show Processed Composition</Typography>
                            <Switch
                                checked={showComposition}
                                onChange={(e) => setShowComposition(e.target.checked)}
                                color="primary"
                                disabled={isDisabled}
                            />
                        </Box>
                    )}
                    <Button
                        variant="contained"
                        size="large"
                        onClick={handlePredict}
                        sx={{ minWidth: 200, py: 1.5, fontSize: '1.1rem' }}
                        disabled={isDisabled}
                    >
                        Predict Properties
                    </Button>
                </Box>

                {error && (
                    <Alert severity="error" sx={{ mt: 3 }}>{error}</Alert>
                )}
            </Paper>

            {prediction && (
                <Box component={motion.div} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                    {showComposition && prediction.composition && (
                        <Paper sx={{ p: 3, mb: 4, bgcolor: 'background.default', border: '1px dashed #ccc' }}>
                            <Typography variant="h6" gutterBottom>Processed Composition:</Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {Object.entries(prediction.composition).map(([el, amt]) => (
                                    <Chip key={el} label={`${el}: ${amt.toFixed(3)}`} variant="outlined" color="primary" />
                                ))}
                            </Box>
                        </Paper>
                    )}

                    <Grid container spacing={4}>
                        <Grid size={{ xs: 12, md: 6 }}>
                            <Paper sx={{ p: 4, textAlign: 'center', bgcolor: '#e3f2fd', border: '1px solid #90caf9' }}>
                                <Typography variant="h6" color="primary" gutterBottom>Piezoelectric Coefficient (d<sub>33</sub>)</Typography>
                                <Typography variant="h2" fontWeight="bold" color="primary.dark">{prediction.d33.toFixed(2)}</Typography>
                                <Typography variant="subtitle1" color="text.secondary">pC/N</Typography>
                            </Paper>
                        </Grid>
                        <Grid size={{ xs: 12, md: 6 }}>
                            <Paper sx={{ p: 4, textAlign: 'center', bgcolor: '#fce4ec', border: '1px solid #f48fb1' }}>
                                <Typography variant="h6" color="secondary" gutterBottom>Curie Temperature (T<sub>c</sub>)</Typography>
                                <Typography variant="h2" fontWeight="bold" color="secondary.dark">{prediction.Tc.toFixed(2)}</Typography>
                                <Typography variant="subtitle1" color="text.secondary">°C</Typography>
                            </Paper>
                        </Grid>
                    </Grid>
                </Box>
            )}
        </Box>
    );
};

export default Prediction;
