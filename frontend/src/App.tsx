import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import Catalog from "./pages/Catalog"
import PDP from "./pages/PDP"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/products" replace />} />
        <Route path="/products" element={<Catalog />} />
        <Route path="/products/:id" element={<PDP />} />
      </Routes>
    </BrowserRouter>
  )
}
