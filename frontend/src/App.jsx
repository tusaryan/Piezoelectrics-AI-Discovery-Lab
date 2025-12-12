import React from 'react';
import { Box, Container } from '@mui/material';
import { motion } from 'framer-motion';
import Layout from './components/Layout';
import Home from './components/Home';
import Prediction from './components/Prediction';
import ModelInsights from './components/ModelInsights';
import Retraining from './components/Retraining';
import DatasetViewer from './components/DatasetViewer';
import ProjectInfo from './components/ProjectInfo';
import useTrainingStatus from './hooks/useTrainingStatus';

const Section = ({ id, children, bgColor = 'transparent' }) => {
  return (
    <Box
      id={id}
      component={motion.div}
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.8 }}
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        py: 8,
        bgcolor: bgColor,
      }}
    >
      <Container maxWidth="xl">
        {children}
      </Container>
    </Box>
  );
};

function App() {
  const [refreshInsights, setRefreshInsights] = React.useState(0);
  const { isTraining, progress, statusMessage, isTrained, logs, error } = useTrainingStatus();

  // Trigger refresh when training completes (isTraining goes from true to false)
  // We can also just rely on the hook's state, but keeping this for explicit refresh triggers if needed
  React.useEffect(() => {
    if (!isTraining && isTrained) {
      setRefreshInsights(prev => prev + 1);
    }
  }, [isTraining, isTrained]);

  return (
    <Layout>
      <Section id="home">
        <Home />
      </Section>

      <Section id="predict" bgColor="#f0f4f8">
        <Prediction isTraining={isTraining} isTrained={isTrained} />
      </Section>

      <Section id="insights">
        <ModelInsights
          refreshTrigger={refreshInsights}
          isTraining={isTraining}
          isTrained={isTrained}
        />
      </Section>

      <Section id="train" bgColor="#f0f4f8">
        <Retraining
          isTraining={isTraining}
          progress={progress}
          statusMessage={statusMessage}
          trainingLogs={logs}
          errorState={error}
        />
      </Section>

      <Section id="dataset">
        <DatasetViewer />
      </Section>

      <Section id="about" bgColor="#f0f4f8">
        <ProjectInfo />
      </Section>
    </Layout>
  );
}

export default App;
