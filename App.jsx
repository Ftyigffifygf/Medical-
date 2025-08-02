import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { 
  Activity, 
  Brain, 
  Heart, 
  Stethoscope, 
  Upload, 
  FileImage, 
  BarChart3, 
  Users, 
  Calendar, 
  Search,
  Shield,
  Zap,
  CheckCircle,
  AlertCircle,
  TrendingUp,
  Camera,
  Microscope,
  Pill,
  FileText,
  Settings,
  User,
  Bell,
  Menu,
  X
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import './App.css'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [analysisResults, setAnalysisResults] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  // Mock data for dashboard
  const dashboardStats = {
    consultations: { value: 47, change: '+9' },
    imagesAnalyzed: { value: 23, change: '+7' },
    evidenceQueries: { value: 15, change: '+3' },
    soapNotes: { value: 31, change: '+1' }
  }

  const recentPatients = [
    { id: 1, name: 'Jane Doe', condition: 'Chest X-ray Analysis', status: 'completed', time: '2 hours ago' },
    { id: 2, name: 'John Smith', condition: 'CT Scan Review', status: 'pending', time: '4 hours ago' },
    { id: 3, name: 'Sarah Wilson', condition: 'MRI Brain Analysis', status: 'completed', time: '6 hours ago' }
  ]

  const handleImageUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleAnalyzeImage = () => {
    setIsAnalyzing(true)
    // Simulate analysis
    setTimeout(() => {
      setAnalysisResults({
        findings: [
          { finding: 'Normal cardiac silhouette', confidence: 92 },
          { finding: 'Clear lung fields bilaterally', confidence: 89 },
          { finding: 'No acute abnormalities', confidence: 95 }
        ],
        impression: 'Normal chest X-ray study',
        recommendations: [
          'Routine follow-up as clinically indicated',
          'No immediate intervention required'
        ]
      })
      setIsAnalyzing(false)
    }, 3000)
  }

  const StatCard = ({ title, value, change, icon: Icon, color }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card className="relative overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            {title}
          </CardTitle>
          <Icon className={`h-4 w-4 ${color}`} />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{value}</div>
          <p className="text-xs text-muted-foreground">
            <span className="text-green-600">{change}</span> from yesterday
          </p>
        </CardContent>
      </Card>
    </motion.div>
  )

  const Navigation = () => (
    <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                <Stethoscope className="h-6 w-6 text-white" />
              </div>
              <span className="ml-3 text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                MedExpert
              </span>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'dashboard' 
                  ? 'text-blue-600 bg-blue-50' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Dashboard
            </button>
            <button
              onClick={() => setActiveTab('imaging')}
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'imaging' 
                  ? 'text-blue-600 bg-blue-50' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Medical Imaging
            </button>
            <button
              onClick={() => setActiveTab('consultation')}
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'consultation' 
                  ? 'text-blue-600 bg-blue-50' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Consultation
            </button>
            <button
              onClick={() => setActiveTab('evidence')}
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === 'evidence' 
                  ? 'text-blue-600 bg-blue-50' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              Evidence
            </button>
          </div>

          <div className="flex items-center space-x-4">
            <Button variant="ghost" size="sm">
              <Bell className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm">
              <Settings className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm">
              <User className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>
    </nav>
  )

  const DashboardContent = () => (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 rounded-2xl p-8 text-white">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-3xl font-bold mb-2">Welcome to MedExpert</h1>
          <p className="text-blue-100 mb-6">
            Advanced AI-powered medical assistant for healthcare professionals
          </p>
          <div className="flex flex-wrap gap-4">
            <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
              <Shield className="w-3 h-3 mr-1" />
              HIPAA Compliant
            </Badge>
            <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
              <Zap className="w-3 h-3 mr-1" />
              MONAI Powered
            </Badge>
            <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
              <CheckCircle className="w-3 h-3 mr-1" />
              FDA Cleared
            </Badge>
          </div>
        </motion.div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Consultations Today"
          value={dashboardStats.consultations.value}
          change={dashboardStats.consultations.change}
          icon={Users}
          color="text-blue-600"
        />
        <StatCard
          title="Images Analyzed"
          value={dashboardStats.imagesAnalyzed.value}
          change={dashboardStats.imagesAnalyzed.change}
          icon={FileImage}
          color="text-green-600"
        />
        <StatCard
          title="Evidence Queries"
          value={dashboardStats.evidenceQueries.value}
          change={dashboardStats.evidenceQueries.change}
          icon={Search}
          color="text-purple-600"
        />
        <StatCard
          title="SOAP Notes"
          value={dashboardStats.soapNotes.value}
          change={dashboardStats.soapNotes.change}
          icon={FileText}
          color="text-orange-600"
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="hover:shadow-lg transition-shadow cursor-pointer" onClick={() => setActiveTab('consultation')}>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Stethoscope className="h-5 w-5 mr-2 text-blue-600" />
              Clinical Consultation
            </CardTitle>
            <CardDescription>
              AI-powered differential diagnosis and clinical decision support
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• Symptom analysis</li>
              <li>• Differential diagnosis</li>
              <li>• Treatment recommendations</li>
            </ul>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow cursor-pointer" onClick={() => setActiveTab('imaging')}>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Camera className="h-5 w-5 mr-2 text-green-600" />
              Medical Imaging
            </CardTitle>
            <CardDescription>
              Advanced medical image analysis with MONAI framework
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• X-ray analysis</li>
              <li>• CT scan interpretation</li>
              <li>• MRI evaluation</li>
            </ul>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow cursor-pointer" onClick={() => setActiveTab('evidence')}>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Microscope className="h-5 w-5 mr-2 text-purple-600" />
              Evidence Synthesis
            </CardTitle>
            <CardDescription>
              Biomedical literature analysis and evidence-based recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• PubMed integration</li>
              <li>• Literature synthesis</li>
              <li>• Clinical guidelines</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Patient Activity</CardTitle>
          <CardDescription>Latest consultations and analyses</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {recentPatients.map((patient) => (
              <div key={patient.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                    <User className="w-4 h-4 text-blue-600" />
                  </div>
                  <div>
                    <p className="font-medium">{patient.name}</p>
                    <p className="text-sm text-muted-foreground">{patient.condition}</p>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant={patient.status === 'completed' ? 'default' : 'secondary'}>
                    {patient.status}
                  </Badge>
                  <p className="text-xs text-muted-foreground mt-1">{patient.time}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const ImagingContent = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Medical Imaging Analysis</h2>
        <p className="text-muted-foreground">
          Upload medical images for AI-powered analysis using MONAI framework
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Upload className="h-5 w-5 mr-2" />
              Upload Medical Image
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              {uploadedImage ? (
                <div className="space-y-4">
                  <img 
                    src={uploadedImage} 
                    alt="Uploaded medical image" 
                    className="max-w-full h-48 object-contain mx-auto rounded-lg"
                  />
                  <Button onClick={() => setUploadedImage(null)} variant="outline" size="sm">
                    Remove Image
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <FileImage className="h-12 w-12 text-gray-400 mx-auto" />
                  <div>
                    <p className="text-sm text-gray-600">
                      Drag and drop your medical image here, or click to browse
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Supports JPEG, PNG, DICOM formats
                    </p>
                  </div>
                  <Input
                    type="file"
                    accept="image/*,.dcm"
                    onChange={handleImageUpload}
                    className="hidden"
                    id="image-upload"
                  />
                  <Label htmlFor="image-upload">
                    <Button variant="outline" className="cursor-pointer">
                      Browse Files
                    </Button>
                  </Label>
                </div>
              )}
            </div>

            {uploadedImage && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="image-type">Image Type</Label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="chest-xray">Chest X-ray</SelectItem>
                        <SelectItem value="ct-scan">CT Scan</SelectItem>
                        <SelectItem value="mri">MRI</SelectItem>
                        <SelectItem value="ultrasound">Ultrasound</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="analysis-type">Analysis Type</Label>
                    <Select>
                      <SelectTrigger>
                        <SelectValue placeholder="Select analysis" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="general">General Analysis</SelectItem>
                        <SelectItem value="pathology">Pathology Detection</SelectItem>
                        <SelectItem value="segmentation">Anatomical Segmentation</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <Button 
                  onClick={handleAnalyzeImage} 
                  className="w-full" 
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Analyzing with MONAI...
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Analyze Image
                    </>
                  )}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="h-5 w-5 mr-2" />
              Analysis Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            {analysisResults ? (
              <div className="space-y-6">
                <div>
                  <h4 className="font-medium mb-3">Findings</h4>
                  <div className="space-y-3">
                    {analysisResults.findings.map((finding, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm">{finding.finding}</span>
                        <div className="flex items-center space-x-2">
                          <Progress value={finding.confidence} className="w-16" />
                          <span className="text-xs text-muted-foreground">{finding.confidence}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Impression</h4>
                  <p className="text-sm text-muted-foreground bg-green-50 p-3 rounded-lg">
                    {analysisResults.impression}
                  </p>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Recommendations</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    {analysisResults.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start">
                        <CheckCircle className="h-4 w-4 text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-muted-foreground">
                  Upload and analyze a medical image to see results here
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )

  const ConsultationContent = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Clinical Consultation</h2>
        <p className="text-muted-foreground">
          AI-powered differential diagnosis and clinical decision support
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Patient Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="age">Age</Label>
              <Input id="age" placeholder="Enter age" />
            </div>
            <div>
              <Label htmlFor="gender">Gender</Label>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Select gender" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="male">Male</SelectItem>
                  <SelectItem value="female">Female</SelectItem>
                  <SelectItem value="other">Other</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="weight">Weight (kg)</Label>
              <Input id="weight" placeholder="Enter weight" />
            </div>
          </div>
          
          <div>
            <Label htmlFor="chief-complaint">Chief Complaint</Label>
            <Textarea 
              id="chief-complaint" 
              placeholder="Describe the patient's primary concern..."
              className="min-h-[100px]"
            />
          </div>

          <div>
            <Label htmlFor="symptoms">Symptoms & History</Label>
            <Textarea 
              id="symptoms" 
              placeholder="List symptoms, duration, severity, and relevant medical history..."
              className="min-h-[120px]"
            />
          </div>

          <Button className="w-full">
            <Activity className="h-4 w-4 mr-2" />
            Generate Differential Diagnosis
          </Button>
        </CardContent>
      </Card>
    </div>
  )

  const EvidenceContent = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Evidence Synthesis</h2>
        <p className="text-muted-foreground">
          Search and analyze biomedical literature for evidence-based recommendations
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Literature Search</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="search-query">Clinical Question</Label>
            <Input 
              id="search-query" 
              placeholder="e.g., What is the efficacy of ACE inhibitors in heart failure?"
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="study-type">Study Type</Label>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="All study types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="rct">Randomized Controlled Trial</SelectItem>
                  <SelectItem value="meta-analysis">Meta-Analysis</SelectItem>
                  <SelectItem value="systematic-review">Systematic Review</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="date-range">Publication Date</Label>
              <Select>
                <SelectTrigger>
                  <SelectValue placeholder="Last 5 years" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1year">Last 1 year</SelectItem>
                  <SelectItem value="5years">Last 5 years</SelectItem>
                  <SelectItem value="10years">Last 10 years</SelectItem>
                  <SelectItem value="all">All time</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button className="w-full">
            <Search className="h-4 w-4 mr-2" />
            Search PubMed & Guidelines
          </Button>
        </CardContent>
      </Card>
    </div>
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      <Navigation />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === 'dashboard' && <DashboardContent />}
            {activeTab === 'imaging' && <ImagingContent />}
            {activeTab === 'consultation' && <ConsultationContent />}
            {activeTab === 'evidence' && <EvidenceContent />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 md:hidden"
          >
            <div className="fixed inset-0 bg-black/50" onClick={() => setIsMenuOpen(false)} />
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              className="fixed right-0 top-0 h-full w-64 bg-white shadow-xl p-6"
            >
              <div className="flex justify-between items-center mb-8">
                <span className="text-lg font-semibold">Menu</span>
                <Button variant="ghost" size="sm" onClick={() => setIsMenuOpen(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <nav className="space-y-4">
                {['dashboard', 'imaging', 'consultation', 'evidence'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => {
                      setActiveTab(tab)
                      setIsMenuOpen(false)
                    }}
                    className={`block w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      activeTab === tab 
                        ? 'text-blue-600 bg-blue-50' 
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </nav>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default App

